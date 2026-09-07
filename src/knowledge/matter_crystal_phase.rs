// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Fail-closed crystal/phase admission for Matter Observatory claims.
//!
//! This layer sits above the generic cross-scale evidence calculus.  It answers a
//! narrower materials-science question: has a computational crystal candidate
//! supplied the minimum *kind* of evidence needed to discuss phase stability?
//!
//! The contract deliberately refuses several common shortcuts:
//!
//! - a negative heuristic composition score is not thermodynamic phase stability;
//! - generic DFT is not automatically periodic-solid electronic structure;
//! - a structure is not "relaxed" merely because an optimizer ran;
//! - molecular vibrations are not automatically a crystal phonon-stability test;
//! - a heavy-element periodic calculation may not splice a separate atomic
//!   relativistic result onto a non-relativistic solid calculation.
//!
//! All evidence admitted here remains computational.  This module does not mint
//! experimental observation, independent replication, verified chronology, or a
//! probability that a proposed crystal exists or can be synthesized.

use std::collections::BTreeSet;
use std::fmt;

use symthaea_epistemic_types::{
    EpistemicContext, MatterClaim, MatterScale, MatterSolverCapability, MatterValidationStage,
};
use symthaea_evidence_plane::external_receipt::ExternalEvidenceBundle;

use super::matter_cross_scale_lineage::AdmittedCrossScaleMatterClaim;
use super::matter_relativistic::{
    AdmittedRelativisticMatterClaim, RelativisticElectronicRequirement,
};
use super::matter_relativistic_evidence::{
    bind_relativistic_execution_references, RelativisticEvidenceBindingError,
    RelativisticExecutionReferenceBinding,
};

/// Exact scientific role played by one crystal evidence object.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CrystalPhaseEvidenceSlot {
    PeriodicElectronicStructure,
    StructuralRelaxation,
    ThermodynamicPhaseCompetition,
    DynamicalStability,
    /// Advisory composition screening only.  This slot is never a substitute for
    /// `ThermodynamicPhaseCompetition`.
    HeuristicCompositionScreening,
}

/// Target identity and explicit relativistic requirement for a crystal candidate.
///
/// Relativistic need is caller-declared rather than inferred from an arbitrary
/// atomic-number threshold.  When it is present, admission requires the periodic
/// crystal evidence itself to be the exact admitted relativistic calculation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CrystalPhaseTarget {
    pub element_atomic_numbers: Vec<u16>,
    pub relativistic_requirement: Option<RelativisticElectronicRequirement>,
}

impl CrystalPhaseTarget {
    pub fn try_new(
        element_atomic_numbers: Vec<u16>,
        relativistic_requirement: Option<RelativisticElectronicRequirement>,
    ) -> Result<Self, CrystalPhaseAdmissionError> {
        if element_atomic_numbers.is_empty() {
            return Err(CrystalPhaseAdmissionError::EmptyElementSet);
        }
        if element_atomic_numbers.iter().any(|z| *z == 0) {
            return Err(CrystalPhaseAdmissionError::InvalidAtomicNumber);
        }
        let mut unique = BTreeSet::new();
        for z in &element_atomic_numbers {
            if !unique.insert(*z) {
                return Err(CrystalPhaseAdmissionError::DuplicateAtomicNumber(*z));
            }
        }
        Ok(Self {
            element_atomic_numbers,
            relativistic_requirement,
        })
    }
}

#[derive(Debug, Clone, Copy)]
enum CrystalPhaseEvidenceKind<'a> {
    PeriodicElectronicStructure {
        claim: &'a MatterClaim,
    },
    StructuralRelaxation {
        claim: &'a MatterClaim,
        max_force_ev_per_angstrom: f64,
        force_tolerance_ev_per_angstrom: f64,
        criterion_evidence_id: &'a str,
    },
    ThermodynamicPhaseCompetition {
        claim: &'a MatterClaim,
        energy_above_hull_ev_per_atom: f64,
        maximum_allowed_above_hull_ev_per_atom: f64,
        competing_phase_count: usize,
        criterion_evidence_id: &'a str,
    },
    DynamicalStability {
        claim: &'a MatterClaim,
        minimum_frequency_thz: f64,
        imaginary_frequency_tolerance_thz: f64,
        sampled_q_point_count: usize,
        criterion_evidence_id: &'a str,
    },
    HeuristicCompositionScreening {
        claim: &'a MatterClaim,
    },
}

/// Validated, typed evidence for one crystal-phase admission slot.
///
/// Fields are private so required physics slots can only be created through the
/// constructors below, which validate scale/context/stage and the declared
/// numerical criterion shape.
#[derive(Debug, Clone, Copy)]
pub struct CrystalPhaseEvidence<'a> {
    kind: CrystalPhaseEvidenceKind<'a>,
}

impl<'a> CrystalPhaseEvidence<'a> {
    pub fn periodic_electronic_structure(
        claim: &'a MatterClaim,
    ) -> Result<Self, CrystalPhaseAdmissionError> {
        validate_required_crystal_physics_claim(claim)?;
        let solver = claim
            .solver
            .as_ref()
            .ok_or_else(|| CrystalPhaseAdmissionError::MissingSolverProfile {
                claim_id: claim.claim_id.clone(),
            })?;
        if !solver.supports(MatterSolverCapability::PeriodicElectronicStructure) {
            return Err(CrystalPhaseAdmissionError::MissingPeriodicElectronicCapability {
                claim_id: claim.claim_id.clone(),
            });
        }
        Ok(Self {
            kind: CrystalPhaseEvidenceKind::PeriodicElectronicStructure { claim },
        })
    }

    pub fn structural_relaxation(
        claim: &'a MatterClaim,
        max_force_ev_per_angstrom: f64,
        force_tolerance_ev_per_angstrom: f64,
        criterion_evidence_id: &'a str,
    ) -> Result<Self, CrystalPhaseAdmissionError> {
        validate_required_crystal_physics_claim(claim)?;
        require_nonempty_criterion(criterion_evidence_id)?;
        require_finite_nonnegative(max_force_ev_per_angstrom, "max force")?;
        require_finite_nonnegative(force_tolerance_ev_per_angstrom, "force tolerance")?;
        Ok(Self {
            kind: CrystalPhaseEvidenceKind::StructuralRelaxation {
                claim,
                max_force_ev_per_angstrom,
                force_tolerance_ev_per_angstrom,
                criterion_evidence_id,
            },
        })
    }

    pub fn thermodynamic_phase_competition(
        claim: &'a MatterClaim,
        energy_above_hull_ev_per_atom: f64,
        maximum_allowed_above_hull_ev_per_atom: f64,
        competing_phase_count: usize,
        criterion_evidence_id: &'a str,
    ) -> Result<Self, CrystalPhaseAdmissionError> {
        validate_required_crystal_physics_claim(claim)?;
        require_nonempty_criterion(criterion_evidence_id)?;
        require_finite_nonnegative(energy_above_hull_ev_per_atom, "energy above hull")?;
        require_finite_nonnegative(
            maximum_allowed_above_hull_ev_per_atom,
            "maximum allowed energy above hull",
        )?;
        if competing_phase_count == 0 {
            return Err(CrystalPhaseAdmissionError::NoCompetingPhasesEvaluated);
        }
        Ok(Self {
            kind: CrystalPhaseEvidenceKind::ThermodynamicPhaseCompetition {
                claim,
                energy_above_hull_ev_per_atom,
                maximum_allowed_above_hull_ev_per_atom,
                competing_phase_count,
                criterion_evidence_id,
            },
        })
    }

    pub fn dynamical_stability(
        claim: &'a MatterClaim,
        minimum_frequency_thz: f64,
        imaginary_frequency_tolerance_thz: f64,
        sampled_q_point_count: usize,
        criterion_evidence_id: &'a str,
    ) -> Result<Self, CrystalPhaseAdmissionError> {
        validate_required_crystal_physics_claim(claim)?;
        require_nonempty_criterion(criterion_evidence_id)?;
        if !minimum_frequency_thz.is_finite() {
            return Err(CrystalPhaseAdmissionError::InvalidMetric("minimum phonon frequency"));
        }
        require_finite_nonnegative(
            imaginary_frequency_tolerance_thz,
            "imaginary-frequency tolerance",
        )?;
        if sampled_q_point_count == 0 {
            return Err(CrystalPhaseAdmissionError::NoPhononQPointsEvaluated);
        }
        Ok(Self {
            kind: CrystalPhaseEvidenceKind::DynamicalStability {
                claim,
                minimum_frequency_thz,
                imaginary_frequency_tolerance_thz,
                sampled_q_point_count,
                criterion_evidence_id,
            },
        })
    }

    /// Mark an advisory composition-screening claim.  Admission records this
    /// evidence but never counts it as phase-competition evidence.
    pub fn heuristic_composition_screening(
        claim: &'a MatterClaim,
    ) -> Result<Self, CrystalPhaseAdmissionError> {
        validate_crystal_claim_identity(claim)?;
        Ok(Self {
            kind: CrystalPhaseEvidenceKind::HeuristicCompositionScreening { claim },
        })
    }

    pub fn slot(&self) -> CrystalPhaseEvidenceSlot {
        match self.kind {
            CrystalPhaseEvidenceKind::PeriodicElectronicStructure { .. } => {
                CrystalPhaseEvidenceSlot::PeriodicElectronicStructure
            }
            CrystalPhaseEvidenceKind::StructuralRelaxation { .. } => {
                CrystalPhaseEvidenceSlot::StructuralRelaxation
            }
            CrystalPhaseEvidenceKind::ThermodynamicPhaseCompetition { .. } => {
                CrystalPhaseEvidenceSlot::ThermodynamicPhaseCompetition
            }
            CrystalPhaseEvidenceKind::DynamicalStability { .. } => {
                CrystalPhaseEvidenceSlot::DynamicalStability
            }
            CrystalPhaseEvidenceKind::HeuristicCompositionScreening { .. } => {
                CrystalPhaseEvidenceSlot::HeuristicCompositionScreening
            }
        }
    }

    pub fn claim(&self) -> &MatterClaim {
        match self.kind {
            CrystalPhaseEvidenceKind::PeriodicElectronicStructure { claim }
            | CrystalPhaseEvidenceKind::StructuralRelaxation { claim, .. }
            | CrystalPhaseEvidenceKind::ThermodynamicPhaseCompetition { claim, .. }
            | CrystalPhaseEvidenceKind::DynamicalStability { claim, .. }
            | CrystalPhaseEvidenceKind::HeuristicCompositionScreening { claim } => claim,
        }
    }

    fn criterion_passes(&self) -> bool {
        match self.kind {
            CrystalPhaseEvidenceKind::PeriodicElectronicStructure { .. }
            | CrystalPhaseEvidenceKind::HeuristicCompositionScreening { .. } => true,
            CrystalPhaseEvidenceKind::StructuralRelaxation {
                max_force_ev_per_angstrom,
                force_tolerance_ev_per_angstrom,
                ..
            } => max_force_ev_per_angstrom <= force_tolerance_ev_per_angstrom,
            CrystalPhaseEvidenceKind::ThermodynamicPhaseCompetition {
                energy_above_hull_ev_per_atom,
                maximum_allowed_above_hull_ev_per_atom,
                ..
            } => energy_above_hull_ev_per_atom <= maximum_allowed_above_hull_ev_per_atom,
            CrystalPhaseEvidenceKind::DynamicalStability {
                minimum_frequency_thz,
                imaginary_frequency_tolerance_thz,
                ..
            } => minimum_frequency_thz >= -imaginary_frequency_tolerance_thz,
        }
    }

    fn snapshot(&self) -> CrystalPhaseEvidenceSnapshot {
        let diagnostic = match self.kind {
            CrystalPhaseEvidenceKind::PeriodicElectronicStructure { .. } => {
                CrystalPhaseDiagnosticSnapshot::PeriodicElectronicStructure
            }
            CrystalPhaseEvidenceKind::StructuralRelaxation {
                max_force_ev_per_angstrom,
                force_tolerance_ev_per_angstrom,
                criterion_evidence_id,
                ..
            } => CrystalPhaseDiagnosticSnapshot::StructuralRelaxation {
                max_force_bits: max_force_ev_per_angstrom.to_bits(),
                force_tolerance_bits: force_tolerance_ev_per_angstrom.to_bits(),
                criterion_evidence_id: criterion_evidence_id.to_string(),
            },
            CrystalPhaseEvidenceKind::ThermodynamicPhaseCompetition {
                energy_above_hull_ev_per_atom,
                maximum_allowed_above_hull_ev_per_atom,
                competing_phase_count,
                criterion_evidence_id,
                ..
            } => CrystalPhaseDiagnosticSnapshot::ThermodynamicPhaseCompetition {
                energy_above_hull_bits: energy_above_hull_ev_per_atom.to_bits(),
                maximum_allowed_above_hull_bits: maximum_allowed_above_hull_ev_per_atom.to_bits(),
                competing_phase_count,
                criterion_evidence_id: criterion_evidence_id.to_string(),
            },
            CrystalPhaseEvidenceKind::DynamicalStability {
                minimum_frequency_thz,
                imaginary_frequency_tolerance_thz,
                sampled_q_point_count,
                criterion_evidence_id,
                ..
            } => CrystalPhaseDiagnosticSnapshot::DynamicalStability {
                minimum_frequency_bits: minimum_frequency_thz.to_bits(),
                imaginary_frequency_tolerance_bits: imaginary_frequency_tolerance_thz.to_bits(),
                sampled_q_point_count,
                criterion_evidence_id: criterion_evidence_id.to_string(),
            },
            CrystalPhaseEvidenceKind::HeuristicCompositionScreening { .. } => {
                CrystalPhaseDiagnosticSnapshot::HeuristicCompositionScreening
            }
        };
        CrystalPhaseEvidenceSnapshot {
            slot: self.slot(),
            claim: self.claim().clone(),
            diagnostic,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CrystalPhaseDiagnosticSnapshot {
    PeriodicElectronicStructure,
    StructuralRelaxation {
        max_force_bits: u64,
        force_tolerance_bits: u64,
        criterion_evidence_id: String,
    },
    ThermodynamicPhaseCompetition {
        energy_above_hull_bits: u64,
        maximum_allowed_above_hull_bits: u64,
        competing_phase_count: usize,
        criterion_evidence_id: String,
    },
    DynamicalStability {
        minimum_frequency_bits: u64,
        imaginary_frequency_tolerance_bits: u64,
        sampled_q_point_count: usize,
        criterion_evidence_id: String,
    },
    HeuristicCompositionScreening,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CrystalPhaseEvidenceSnapshot {
    pub slot: CrystalPhaseEvidenceSlot,
    pub claim: MatterClaim,
    pub diagnostic: CrystalPhaseDiagnosticSnapshot,
}

/// Fully admitted computational crystal candidate with all upstream and
/// crystal-specific evidence retained.
#[derive(Debug, Clone, PartialEq)]
pub struct AdmittedCrystalPhaseMatterClaim {
    pub target: CrystalPhaseTarget,
    pub upstream: AdmittedCrossScaleMatterClaim,
    pub crystal_evidence: Vec<CrystalPhaseEvidenceSnapshot>,
    pub relativistic_claim: Option<AdmittedRelativisticMatterClaim>,
    pub relativistic_reference_binding: Option<RelativisticExecutionReferenceBinding>,
}

/// Admit a computational crystal candidate only when all required crystal-physics
/// slots are present and satisfy their declared criteria.
pub fn admit_crystal_phase_candidate(
    upstream: AdmittedCrossScaleMatterClaim,
    target: CrystalPhaseTarget,
    evidence: &[CrystalPhaseEvidence<'_>],
    relativistic_claim: Option<&AdmittedRelativisticMatterClaim>,
    external_evidence: Option<&ExternalEvidenceBundle>,
) -> Result<AdmittedCrystalPhaseMatterClaim, CrystalPhaseAdmissionError> {
    if upstream.claim.scale != MatterScale::Crystal {
        return Err(CrystalPhaseAdmissionError::UpstreamClaimIsNotCrystal);
    }

    let mut seen_slots = BTreeSet::new();
    let mut seen_claim_ids = BTreeSet::new();
    let mut snapshots = Vec::with_capacity(evidence.len());
    let mut saw_heuristic_screening = false;
    let mut periodic_claim_id: Option<String> = None;
    let mut crystal_ood = false;

    for item in evidence {
        let slot = item.slot();
        let claim = item.claim();
        if !seen_claim_ids.insert(claim.claim_id.as_str()) {
            return Err(CrystalPhaseAdmissionError::DuplicateEvidenceClaimId(
                claim.claim_id.clone(),
            ));
        }

        if slot == CrystalPhaseEvidenceSlot::HeuristicCompositionScreening {
            saw_heuristic_screening = true;
        } else if !seen_slots.insert(slot) {
            return Err(CrystalPhaseAdmissionError::DuplicateRequiredEvidenceSlot(slot));
        }

        if !item.criterion_passes() {
            return Err(CrystalPhaseAdmissionError::DeclaredCriterionNotMet(slot));
        }
        if claim
            .uncertainty
            .as_ref()
            .is_some_and(|uncertainty| uncertainty.out_of_domain)
        {
            crystal_ood = true;
        }
        if slot == CrystalPhaseEvidenceSlot::PeriodicElectronicStructure {
            periodic_claim_id = Some(claim.claim_id.clone());
        }
        snapshots.push(item.snapshot());
    }

    for required in [
        CrystalPhaseEvidenceSlot::PeriodicElectronicStructure,
        CrystalPhaseEvidenceSlot::StructuralRelaxation,
        CrystalPhaseEvidenceSlot::ThermodynamicPhaseCompetition,
        CrystalPhaseEvidenceSlot::DynamicalStability,
    ] {
        if !seen_slots.contains(&required) {
            if required == CrystalPhaseEvidenceSlot::ThermodynamicPhaseCompetition
                && saw_heuristic_screening
            {
                return Err(
                    CrystalPhaseAdmissionError::HeuristicScreeningCannotEstablishPhaseStability,
                );
            }
            return Err(CrystalPhaseAdmissionError::MissingRequiredEvidenceSlot(
                required,
            ));
        }
    }

    if crystal_ood
        && !upstream
            .claim
            .uncertainty
            .as_ref()
            .is_some_and(|uncertainty| uncertainty.out_of_domain)
    {
        return Err(CrystalPhaseAdmissionError::CrystalOodStatusNotPropagated);
    }

    let (relativistic_claim_snapshot, relativistic_binding) =
        if let Some(required) = target.relativistic_requirement {
            let admitted = relativistic_claim
                .ok_or(CrystalPhaseAdmissionError::MissingRelativisticCrystalEvidence)?;
            if admitted.claim.scale != MatterScale::Crystal {
                return Err(CrystalPhaseAdmissionError::RelativisticClaimIsNotCrystal);
            }
            if admitted.receipt.requirement != required {
                return Err(CrystalPhaseAdmissionError::RelativisticRequirementMismatch {
                    expected: required,
                    actual: admitted.receipt.requirement,
                });
            }

            let target_elements: BTreeSet<u16> =
                target.element_atomic_numbers.iter().copied().collect();
            let receipt_elements: BTreeSet<u16> =
                admitted.receipt.element_atomic_numbers.iter().copied().collect();
            if target_elements != receipt_elements {
                return Err(CrystalPhaseAdmissionError::RelativisticElementSetMismatch);
            }
            if !admitted
                .receipt
                .solver
                .supports(MatterSolverCapability::PeriodicElectronicStructure)
            {
                return Err(
                    CrystalPhaseAdmissionError::RelativisticCrystalLacksPeriodicCapability,
                );
            }
            if periodic_claim_id.as_deref() != Some(admitted.claim.claim_id.as_str()) {
                return Err(
                    CrystalPhaseAdmissionError::PeriodicEvidenceMustBeRelativisticCrystalClaim,
                );
            }

            let bundle = external_evidence
                .ok_or(CrystalPhaseAdmissionError::MissingRelativisticExternalEvidence)?;
            let binding = bind_relativistic_execution_references(&admitted.receipt, bundle)?;
            (Some(admitted.clone()), Some(binding))
        } else {
            (None, None)
        };

    Ok(AdmittedCrystalPhaseMatterClaim {
        target,
        upstream,
        crystal_evidence: snapshots,
        relativistic_claim: relativistic_claim_snapshot,
        relativistic_reference_binding: relativistic_binding,
    })
}

fn validate_crystal_claim_identity(claim: &MatterClaim) -> Result<(), CrystalPhaseAdmissionError> {
    if claim.claim_id.trim().is_empty() {
        return Err(CrystalPhaseAdmissionError::EmptyEvidenceClaimId);
    }
    if !matches!(claim.scale, MatterScale::Crystal | MatterScale::Multiscale) {
        return Err(CrystalPhaseAdmissionError::EvidenceClaimScaleMismatch {
            claim_id: claim.claim_id.clone(),
            actual: claim.scale,
        });
    }
    if claim.epistemic.context != EpistemicContext::Scientific {
        return Err(CrystalPhaseAdmissionError::NonScientificEvidenceClaim {
            claim_id: claim.claim_id.clone(),
        });
    }
    Ok(())
}

fn validate_required_crystal_physics_claim(
    claim: &MatterClaim,
) -> Result<(), CrystalPhaseAdmissionError> {
    validate_crystal_claim_identity(claim)?;
    if !matches!(
        claim.validation_stage,
        MatterValidationStage::PhysicsSimulated
            | MatterValidationStage::CrossModelSupported
            | MatterValidationStage::HighFidelitySimulated
            | MatterValidationStage::ReferenceBenchmarked
    ) {
        return Err(CrystalPhaseAdmissionError::InsufficientPhysicsEvidenceStage {
            claim_id: claim.claim_id.clone(),
            stage: claim.validation_stage,
        });
    }
    Ok(())
}

fn require_nonempty_criterion(id: &str) -> Result<(), CrystalPhaseAdmissionError> {
    if id.trim().is_empty() {
        Err(CrystalPhaseAdmissionError::EmptyCriterionEvidenceId)
    } else {
        Ok(())
    }
}

fn require_finite_nonnegative(
    value: f64,
    label: &'static str,
) -> Result<(), CrystalPhaseAdmissionError> {
    if !value.is_finite() || value < 0.0 {
        Err(CrystalPhaseAdmissionError::InvalidMetric(label))
    } else {
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum CrystalPhaseAdmissionError {
    EmptyElementSet,
    InvalidAtomicNumber,
    DuplicateAtomicNumber(u16),
    UpstreamClaimIsNotCrystal,
    EmptyEvidenceClaimId,
    EvidenceClaimScaleMismatch {
        claim_id: String,
        actual: MatterScale,
    },
    NonScientificEvidenceClaim {
        claim_id: String,
    },
    InsufficientPhysicsEvidenceStage {
        claim_id: String,
        stage: MatterValidationStage,
    },
    MissingSolverProfile {
        claim_id: String,
    },
    MissingPeriodicElectronicCapability {
        claim_id: String,
    },
    EmptyCriterionEvidenceId,
    InvalidMetric(&'static str),
    NoCompetingPhasesEvaluated,
    NoPhononQPointsEvaluated,
    DuplicateEvidenceClaimId(String),
    DuplicateRequiredEvidenceSlot(CrystalPhaseEvidenceSlot),
    MissingRequiredEvidenceSlot(CrystalPhaseEvidenceSlot),
    DeclaredCriterionNotMet(CrystalPhaseEvidenceSlot),
    HeuristicScreeningCannotEstablishPhaseStability,
    CrystalOodStatusNotPropagated,
    MissingRelativisticCrystalEvidence,
    MissingRelativisticExternalEvidence,
    RelativisticClaimIsNotCrystal,
    RelativisticRequirementMismatch {
        expected: RelativisticElectronicRequirement,
        actual: RelativisticElectronicRequirement,
    },
    RelativisticElementSetMismatch,
    RelativisticCrystalLacksPeriodicCapability,
    PeriodicEvidenceMustBeRelativisticCrystalClaim,
    RelativisticEvidence(RelativisticEvidenceBindingError),
}

impl From<RelativisticEvidenceBindingError> for CrystalPhaseAdmissionError {
    fn from(value: RelativisticEvidenceBindingError) -> Self {
        Self::RelativisticEvidence(value)
    }
}

impl fmt::Display for CrystalPhaseAdmissionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyElementSet => write!(f, "crystal target element set must not be empty"),
            Self::InvalidAtomicNumber => write!(f, "atomic number zero is invalid"),
            Self::DuplicateAtomicNumber(z) => write!(f, "element atomic number Z={z} is duplicated"),
            Self::UpstreamClaimIsNotCrystal => write!(f, "crystal admission requires an upstream Crystal-scale claim"),
            Self::EmptyEvidenceClaimId => write!(f, "crystal evidence claim id must not be empty"),
            Self::EvidenceClaimScaleMismatch { claim_id, actual } => write!(f, "crystal evidence claim `{claim_id}` addresses {actual:?}, expected Crystal or Multiscale"),
            Self::NonScientificEvidenceClaim { claim_id } => write!(f, "crystal evidence claim `{claim_id}` is not in Scientific epistemic context"),
            Self::InsufficientPhysicsEvidenceStage { claim_id, stage } => write!(f, "crystal evidence claim `{claim_id}` has {stage:?}; required physics slots cannot be satisfied by hypothesis/surrogate/experimental-label shortcuts"),
            Self::MissingSolverProfile { claim_id } => write!(f, "periodic electronic evidence claim `{claim_id}` has no solver profile"),
            Self::MissingPeriodicElectronicCapability { claim_id } => write!(f, "claim `{claim_id}` does not explicitly advertise PeriodicElectronicStructure"),
            Self::EmptyCriterionEvidenceId => write!(f, "crystal numerical criterion evidence id must not be empty"),
            Self::InvalidMetric(label) => write!(f, "crystal metric `{label}` must be finite and physically admissible"),
            Self::NoCompetingPhasesEvaluated => write!(f, "phase-competition evidence must evaluate at least one competing phase"),
            Self::NoPhononQPointsEvaluated => write!(f, "dynamical-stability evidence must sample at least one q-point"),
            Self::DuplicateEvidenceClaimId(id) => write!(f, "crystal evidence claim `{id}` is reused across more than one evidence object"),
            Self::DuplicateRequiredEvidenceSlot(slot) => write!(f, "required crystal evidence slot {slot:?} occurs more than once"),
            Self::MissingRequiredEvidenceSlot(slot) => write!(f, "required crystal evidence slot {slot:?} is missing"),
            Self::DeclaredCriterionNotMet(slot) => write!(f, "crystal evidence in slot {slot:?} violates its declared numerical criterion"),
            Self::HeuristicScreeningCannotEstablishPhaseStability => write!(f, "heuristic composition screening cannot satisfy thermodynamic phase-competition evidence"),
            Self::CrystalOodStatusNotPropagated => write!(f, "OOD crystal evidence must propagate an OOD uncertainty marker to the upstream crystal claim"),
            Self::MissingRelativisticCrystalEvidence => write!(f, "target requires relativistic crystal evidence but none was supplied"),
            Self::MissingRelativisticExternalEvidence => write!(f, "relativistic crystal admission requires the canonical external evidence bundle so the execution binding can be revalidated"),
            Self::RelativisticClaimIsNotCrystal => write!(f, "relativistic evidence used for a crystal target must itself be a Crystal-scale claim"),
            Self::RelativisticRequirementMismatch { expected, actual } => write!(f, "relativistic crystal requirement mismatch: expected {expected:?}, got {actual:?}"),
            Self::RelativisticElementSetMismatch => write!(f, "relativistic crystal evidence element set does not exactly match the target element set"),
            Self::RelativisticCrystalLacksPeriodicCapability => write!(f, "relativistic crystal solver must explicitly advertise PeriodicElectronicStructure"),
            Self::PeriodicEvidenceMustBeRelativisticCrystalClaim => write!(f, "when relativistic treatment is required, the periodic electronic-structure evidence must be the same admitted relativistic Crystal claim"),
            Self::RelativisticEvidence(error) => write!(f, "relativistic execution evidence binding failed: {error}"),
        }
    }
}

impl std::error::Error for CrystalPhaseAdmissionError {}

/// Explicit interpretation of the current `symthaea-materials` composition model.
#[cfg(feature = "materials")]
#[derive(Debug, Clone, PartialEq)]
pub struct AdvisoryCompoundStabilityScreening {
    pub formula: String,
    pub formation_energy_proxy_ev_per_atom: f64,
    pub reported_confidence_proxy: f64,
    pub reported_stability_flag: bool,
    pub interpretation: &'static str,
}

/// Convert the current simplified composition predictor into an explicitly
/// advisory record.  This function does not create thermodynamic phase evidence.
#[cfg(feature = "materials")]
pub fn classify_current_compound_stability_screening(
    prediction: &symthaea_materials::compound_stability::StabilityPrediction,
) -> AdvisoryCompoundStabilityScreening {
    AdvisoryCompoundStabilityScreening {
        formula: prediction.formula.clone(),
        formation_energy_proxy_ev_per_atom: prediction.formation_energy,
        reported_confidence_proxy: prediction.confidence,
        reported_stability_flag: prediction.is_stable,
        interpretation: "heuristic composition screening only; not thermodynamic phase stability",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::matter_cross_scale::{
        MatterScaleTransition, MatterTransitionSupport,
    };
    use crate::knowledge::matter_cross_scale_lineage::CrossScaleLineageEnvelope;
    use symthaea_epistemic_types::{
        MatterSolverProfile, MatterUncertainty,
    };

    fn solver(name: &str, capabilities: Vec<MatterSolverCapability>) -> MatterSolverProfile {
        MatterSolverProfile {
            name: name.to_string(),
            version: "1".to_string(),
            method: format!("{name} fixture"),
            capabilities,
            limitations: vec!["fixture only".to_string()],
        }
    }

    fn claim(
        id: &str,
        scale: MatterScale,
        capabilities: Vec<MatterSolverCapability>,
    ) -> MatterClaim {
        MatterClaim::local_computational(
            id,
            format!("statement for {id}"),
            scale,
            MatterValidationStage::HighFidelitySimulated,
            solver(id, capabilities),
        )
        .unwrap()
    }

    fn admitted_crystal_candidate(ood: bool) -> AdmittedCrossScaleMatterClaim {
        let source = claim("electronic-source", MatterScale::AtomicElectronic, vec![]);
        let mut crystal_transition = claim("electronic-to-crystal", MatterScale::Crystal, vec![]);
        if ood {
            crystal_transition = crystal_transition.with_uncertainty(
                MatterUncertainty::new(1.0, "distance", "fixture OOD", true).unwrap(),
            );
        }
        let transition = if ood {
            MatterScaleTransition::supported_out_of_domain(
                "electronic->crystal",
                MatterScale::AtomicElectronic,
                MatterScale::Crystal,
                &crystal_transition,
            )
        } else {
            MatterScaleTransition::supported_in_domain(
                "electronic->crystal",
                MatterScale::AtomicElectronic,
                MatterScale::Crystal,
                &crystal_transition,
            )
        };
        let envelope = CrossScaleLineageEnvelope::assess(
            &[&source],
            &[transition],
            MatterScale::Crystal,
        )
        .unwrap();
        let mut candidate = claim("crystal-candidate", MatterScale::Crystal, vec![]);
        if ood {
            candidate = candidate.with_uncertainty(
                MatterUncertainty::new(1.0, "distance", "fixture OOD", true).unwrap(),
            );
        }
        envelope.admit_derived_computational_claim(candidate).unwrap()
    }

    fn required_evidence<'a>(
        periodic: &'a MatterClaim,
        relaxation: &'a MatterClaim,
        phase: &'a MatterClaim,
        dynamic: &'a MatterClaim,
    ) -> Vec<CrystalPhaseEvidence<'a>> {
        vec![
            CrystalPhaseEvidence::periodic_electronic_structure(periodic).unwrap(),
            CrystalPhaseEvidence::structural_relaxation(
                relaxation,
                0.01,
                0.02,
                "criterion:relax:v1",
            )
            .unwrap(),
            CrystalPhaseEvidence::thermodynamic_phase_competition(
                phase,
                0.01,
                0.05,
                12,
                "criterion:hull:v1",
            )
            .unwrap(),
            CrystalPhaseEvidence::dynamical_stability(
                dynamic,
                -0.01,
                0.02,
                64,
                "criterion:phonon:v1",
            )
            .unwrap(),
        ]
    }

    #[test]
    fn complete_nonrelativistic_crystal_evidence_admits() {
        let upstream = admitted_crystal_candidate(false);
        let periodic = claim(
            "periodic",
            MatterScale::Crystal,
            vec![MatterSolverCapability::PeriodicElectronicStructure],
        );
        let relaxation = claim("relaxation", MatterScale::Crystal, vec![]);
        let phase = claim("phase", MatterScale::Crystal, vec![]);
        let dynamic = claim("dynamic", MatterScale::Crystal, vec![]);
        let evidence = required_evidence(&periodic, &relaxation, &phase, &dynamic);
        let target = CrystalPhaseTarget::try_new(vec![14, 8], None).unwrap();

        let admitted = admit_crystal_phase_candidate(upstream, target, &evidence, None, None)
            .unwrap();
        assert_eq!(admitted.crystal_evidence.len(), 4);
        assert!(admitted.relativistic_claim.is_none());
    }

    #[test]
    fn periodic_slot_requires_exact_periodic_capability() {
        let generic_dft = claim(
            "generic-dft",
            MatterScale::Crystal,
            vec![MatterSolverCapability::DensityFunctionalTheory],
        );
        assert!(matches!(
            CrystalPhaseEvidence::periodic_electronic_structure(&generic_dft),
            Err(CrystalPhaseAdmissionError::MissingPeriodicElectronicCapability { .. })
        ));
    }

    #[test]
    fn relaxation_must_meet_declared_force_criterion() {
        let evidence_claim = claim("relaxation", MatterScale::Crystal, vec![]);
        let evidence = CrystalPhaseEvidence::structural_relaxation(
            &evidence_claim,
            0.03,
            0.02,
            "criterion:relax:v1",
        )
        .unwrap();
        assert!(!evidence.criterion_passes());
    }

    #[test]
    fn phase_competition_requires_competing_phases() {
        let evidence_claim = claim("phase", MatterScale::Crystal, vec![]);
        assert_eq!(
            CrystalPhaseEvidence::thermodynamic_phase_competition(
                &evidence_claim,
                0.0,
                0.05,
                0,
                "criterion:hull:v1",
            )
            .unwrap_err(),
            CrystalPhaseAdmissionError::NoCompetingPhasesEvaluated
        );
    }

    #[test]
    fn dynamical_stability_requires_q_point_sampling() {
        let evidence_claim = claim("dynamic", MatterScale::Crystal, vec![]);
        assert_eq!(
            CrystalPhaseEvidence::dynamical_stability(
                &evidence_claim,
                0.0,
                0.02,
                0,
                "criterion:phonon:v1",
            )
            .unwrap_err(),
            CrystalPhaseAdmissionError::NoPhononQPointsEvaluated
        );
    }

    #[test]
    fn heuristic_screening_cannot_replace_phase_competition() {
        let upstream = admitted_crystal_candidate(false);
        let periodic = claim(
            "periodic",
            MatterScale::Crystal,
            vec![MatterSolverCapability::PeriodicElectronicStructure],
        );
        let relaxation = claim("relaxation", MatterScale::Crystal, vec![]);
        let dynamic = claim("dynamic", MatterScale::Crystal, vec![]);
        let heuristic = MatterClaim::local_computational(
            "heuristic-screen",
            "simplified composition screen",
            MatterScale::Crystal,
            MatterValidationStage::SurrogateSupported,
            solver("heuristic", vec![]),
        )
        .unwrap();
        let evidence = vec![
            CrystalPhaseEvidence::periodic_electronic_structure(&periodic).unwrap(),
            CrystalPhaseEvidence::structural_relaxation(
                &relaxation,
                0.01,
                0.02,
                "criterion:relax:v1",
            )
            .unwrap(),
            CrystalPhaseEvidence::dynamical_stability(
                &dynamic,
                0.1,
                0.02,
                64,
                "criterion:phonon:v1",
            )
            .unwrap(),
            CrystalPhaseEvidence::heuristic_composition_screening(&heuristic).unwrap(),
        ];
        let target = CrystalPhaseTarget::try_new(vec![11, 17], None).unwrap();
        assert_eq!(
            admit_crystal_phase_candidate(upstream, target, &evidence, None, None).unwrap_err(),
            CrystalPhaseAdmissionError::HeuristicScreeningCannotEstablishPhaseStability
        );
    }

    #[test]
    fn crystal_evidence_ood_must_propagate() {
        let upstream = admitted_crystal_candidate(false);
        let periodic = claim(
            "periodic",
            MatterScale::Crystal,
            vec![MatterSolverCapability::PeriodicElectronicStructure],
        );
        let relaxation = claim("relaxation", MatterScale::Crystal, vec![]);
        let phase = claim("phase", MatterScale::Crystal, vec![]);
        let dynamic = claim("dynamic", MatterScale::Crystal, vec![]).with_uncertainty(
            MatterUncertainty::new(1.0, "distance", "phonon OOD", true).unwrap(),
        );
        let evidence = required_evidence(&periodic, &relaxation, &phase, &dynamic);
        let target = CrystalPhaseTarget::try_new(vec![14, 8], None).unwrap();
        assert_eq!(
            admit_crystal_phase_candidate(upstream, target, &evidence, None, None).unwrap_err(),
            CrystalPhaseAdmissionError::CrystalOodStatusNotPropagated
        );
    }

    #[cfg(feature = "materials")]
    #[test]
    fn current_compound_stability_flag_is_advisory_not_phase_evidence() {
        let prediction = symthaea_materials::compound_stability::predict_stability(
            &[(11, 0.5), (17, 0.5)],
            300.0,
        );
        assert!(prediction.is_stable);
        let advisory = classify_current_compound_stability_screening(&prediction);
        assert!(advisory.reported_stability_flag);
        assert_eq!(
            advisory.interpretation,
            "heuristic composition screening only; not thermodynamic phase stability"
        );
    }
}
