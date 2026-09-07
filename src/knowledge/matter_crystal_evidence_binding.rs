// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Strict canonical evidence binding for admitted crystal/phase claims.
//!
//! `matter_crystal_phase` establishes the computational physics gate. This
//! module is the single public binding boundary above it. It revalidates target
//! identity, cross-scale authority/lineage, crystal evidence shape, stored
//! numerical criteria, solver provenance, OOD propagation, and relativistic
//! execution identity before resolving each numerical criterion through a
//! canonical `Preregistration` reference.
//!
//! A preregistration role remains reference-only. This layer does not prove when
//! a criterion was registered, authenticate its issuer, fetch/re-hash artifact
//! bytes, validate solver numerics, establish synthesis, or upgrade E/N/M.

use std::cmp::min;
use std::collections::BTreeSet;
use std::fmt;

use symthaea_epistemic_types::{
    EpistemicContext, MatterClaim, MatterScale, MatterSolverCapability, MatterValidationStage,
};
use symthaea_evidence_plane::external_receipt::{
    ClaimedUtcDate, EvidenceReferenceInterpretation, EvidenceRole, ExternalEvidenceBundle,
    ExternalEvidenceError, Sha256Digest,
};

use super::matter_cross_scale_lineage::CrossScaleDependencyRole;
use super::matter_crystal_phase::{
    AdmittedCrystalPhaseMatterClaim, CrystalPhaseDiagnosticSnapshot, CrystalPhaseEvidenceSlot,
};
use super::matter_relativistic_evidence::{
    bind_relativistic_execution_references, RelativisticExecutionReferenceBinding,
};

/// Exact canonical preregistration reference attached to one numerical crystal
/// criterion.
///
/// Multiple slots may intentionally point to the same preregistration artifact
/// when one protocol declares several thresholds. No independence is inferred
/// from the number of bindings.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CrystalCriterionReferenceBinding {
    pub slot: CrystalPhaseEvidenceSlot,
    pub evidence_id: String,
    pub content_sha256: Sha256Digest,
    pub claimed_date: Option<ClaimedUtcDate>,
    pub reference_interpretation: EvidenceReferenceInterpretation,
}

/// Crystal admission plus canonical references for every required numerical
/// criterion.
#[derive(Debug, Clone, PartialEq)]
pub struct EvidenceBoundCrystalPhaseMatterClaim {
    pub admitted: AdmittedCrystalPhaseMatterClaim,
    pub criterion_bindings: Vec<CrystalCriterionReferenceBinding>,
    pub reference_interpretation: EvidenceReferenceInterpretation,
}

/// Revalidate an admitted crystal-phase object and bind its relaxation/hull/
/// phonon criteria to canonical `Preregistration` references.
///
/// Revalidation is intentional because the preceding admitted structs expose
/// public fields for inspection and can therefore be hand-constructed or
/// mutated after admission. This function is the public fail-closed boundary.
pub fn bind_crystal_phase_evidence(
    admitted: AdmittedCrystalPhaseMatterClaim,
    bundle: &ExternalEvidenceBundle,
) -> Result<EvidenceBoundCrystalPhaseMatterClaim, CrystalEvidenceBindingError> {
    validate_target_identity(&admitted)?;
    validate_upstream_cross_scale(&admitted)?;

    let mut seen_slots = BTreeSet::new();
    let mut seen_claim_ids = BTreeSet::new();
    let mut criterion_bindings = Vec::new();
    let mut periodic_claim: Option<MatterClaim> = None;
    let mut required_evidence_is_ood = false;

    for snapshot in &admitted.crystal_evidence {
        let claim = &snapshot.claim;
        validate_claim_identity(claim)?;

        if !seen_claim_ids.insert(claim.claim_id.clone()) {
            return Err(CrystalEvidenceBindingError::DuplicateEvidenceClaimId(
                claim.claim_id.clone(),
            ));
        }

        if snapshot.slot != CrystalPhaseEvidenceSlot::HeuristicCompositionScreening {
            if !seen_slots.insert(snapshot.slot) {
                return Err(CrystalEvidenceBindingError::DuplicateRequiredEvidenceSlot(
                    snapshot.slot,
                ));
            }
            validate_required_physics_claim(claim)?;
            if claim
                .uncertainty
                .as_ref()
                .is_some_and(|uncertainty| uncertainty.out_of_domain)
            {
                required_evidence_is_ood = true;
            }
        }

        match (snapshot.slot, &snapshot.diagnostic) {
            (
                CrystalPhaseEvidenceSlot::PeriodicElectronicStructure,
                CrystalPhaseDiagnosticSnapshot::PeriodicElectronicStructure,
            ) => {
                let solver = claim
                    .solver
                    .as_ref()
                    .expect("required solver provenance validated above");
                if !solver.supports(MatterSolverCapability::PeriodicElectronicStructure) {
                    return Err(
                        CrystalEvidenceBindingError::PeriodicClaimLacksPeriodicCapability {
                            claim_id: claim.claim_id.clone(),
                        },
                    );
                }
                periodic_claim = Some(claim.clone());
            }
            (
                CrystalPhaseEvidenceSlot::StructuralRelaxation,
                CrystalPhaseDiagnosticSnapshot::StructuralRelaxation {
                    max_force_bits,
                    force_tolerance_bits,
                    criterion_evidence_id,
                },
            ) => {
                let max_force = f64::from_bits(*max_force_bits);
                let tolerance = f64::from_bits(*force_tolerance_bits);
                require_finite_nonnegative(max_force, "max force")?;
                require_finite_nonnegative(tolerance, "force tolerance")?;
                if max_force > tolerance {
                    return Err(CrystalEvidenceBindingError::StoredCriterionNotMet(
                        snapshot.slot,
                    ));
                }
                criterion_bindings.push(bind_preregistration(
                    snapshot.slot,
                    criterion_evidence_id,
                    bundle,
                )?);
            }
            (
                CrystalPhaseEvidenceSlot::ThermodynamicPhaseCompetition,
                CrystalPhaseDiagnosticSnapshot::ThermodynamicPhaseCompetition {
                    energy_above_hull_bits,
                    maximum_allowed_above_hull_bits,
                    competing_phase_count,
                    criterion_evidence_id,
                },
            ) => {
                let energy_above_hull = f64::from_bits(*energy_above_hull_bits);
                let maximum_allowed = f64::from_bits(*maximum_allowed_above_hull_bits);
                require_finite_nonnegative(energy_above_hull, "energy above hull")?;
                require_finite_nonnegative(
                    maximum_allowed,
                    "maximum allowed energy above hull",
                )?;
                if *competing_phase_count == 0 {
                    return Err(CrystalEvidenceBindingError::NoCompetingPhasesEvaluated);
                }
                if energy_above_hull > maximum_allowed {
                    return Err(CrystalEvidenceBindingError::StoredCriterionNotMet(
                        snapshot.slot,
                    ));
                }
                criterion_bindings.push(bind_preregistration(
                    snapshot.slot,
                    criterion_evidence_id,
                    bundle,
                )?);
            }
            (
                CrystalPhaseEvidenceSlot::DynamicalStability,
                CrystalPhaseDiagnosticSnapshot::DynamicalStability {
                    minimum_frequency_bits,
                    imaginary_frequency_tolerance_bits,
                    sampled_q_point_count,
                    criterion_evidence_id,
                },
            ) => {
                let minimum_frequency = f64::from_bits(*minimum_frequency_bits);
                let tolerance = f64::from_bits(*imaginary_frequency_tolerance_bits);
                if !minimum_frequency.is_finite() {
                    return Err(CrystalEvidenceBindingError::InvalidMetric(
                        "minimum phonon frequency",
                    ));
                }
                require_finite_nonnegative(tolerance, "imaginary-frequency tolerance")?;
                if *sampled_q_point_count == 0 {
                    return Err(CrystalEvidenceBindingError::NoPhononQPointsEvaluated);
                }
                if minimum_frequency < -tolerance {
                    return Err(CrystalEvidenceBindingError::StoredCriterionNotMet(
                        snapshot.slot,
                    ));
                }
                criterion_bindings.push(bind_preregistration(
                    snapshot.slot,
                    criterion_evidence_id,
                    bundle,
                )?);
            }
            (
                CrystalPhaseEvidenceSlot::HeuristicCompositionScreening,
                CrystalPhaseDiagnosticSnapshot::HeuristicCompositionScreening,
            ) => {}
            _ => {
                return Err(CrystalEvidenceBindingError::SlotDiagnosticMismatch(
                    snapshot.slot,
                ));
            }
        }
    }

    for required in [
        CrystalPhaseEvidenceSlot::PeriodicElectronicStructure,
        CrystalPhaseEvidenceSlot::StructuralRelaxation,
        CrystalPhaseEvidenceSlot::ThermodynamicPhaseCompetition,
        CrystalPhaseEvidenceSlot::DynamicalStability,
    ] {
        if !seen_slots.contains(&required) {
            return Err(CrystalEvidenceBindingError::MissingRequiredEvidenceSlot(
                required,
            ));
        }
    }

    if criterion_bindings.len() != 3 {
        return Err(CrystalEvidenceBindingError::IncompleteCriterionBindingSet);
    }

    if required_evidence_is_ood
        && !admitted
            .upstream
            .claim
            .uncertainty
            .as_ref()
            .is_some_and(|uncertainty| uncertainty.out_of_domain)
    {
        return Err(CrystalEvidenceBindingError::CrystalOodStatusNotPropagated);
    }

    revalidate_relativistic_shape(&admitted, periodic_claim.as_ref(), bundle)?;

    Ok(EvidenceBoundCrystalPhaseMatterClaim {
        admitted,
        criterion_bindings,
        reference_interpretation: EvidenceReferenceInterpretation::ReferenceOnly,
    })
}

fn validate_target_identity(
    admitted: &AdmittedCrystalPhaseMatterClaim,
) -> Result<(), CrystalEvidenceBindingError> {
    let elements = &admitted.target.element_atomic_numbers;
    if elements.is_empty() {
        return Err(CrystalEvidenceBindingError::EmptyTargetElementSet);
    }
    if elements.iter().any(|z| *z == 0) {
        return Err(CrystalEvidenceBindingError::InvalidTargetAtomicNumber);
    }
    let unique: BTreeSet<u16> = elements.iter().copied().collect();
    if unique.len() != elements.len() {
        return Err(CrystalEvidenceBindingError::DuplicateTargetAtomicNumber);
    }
    Ok(())
}

fn validate_upstream_cross_scale(
    admitted: &AdmittedCrystalPhaseMatterClaim,
) -> Result<(), CrystalEvidenceBindingError> {
    let upstream = &admitted.upstream;
    if upstream.claim.scale != MatterScale::Crystal {
        return Err(CrystalEvidenceBindingError::UpstreamClaimIsNotCrystal);
    }
    if !upstream.lineage.assessment.is_admissible()
        || !upstream.lineage.assessment.blocking_transition_ids.is_empty()
    {
        return Err(CrystalEvidenceBindingError::UpstreamAssessmentRefused);
    }

    upstream
        .lineage
        .assessment
        .enforce_derived_computational_claim(&upstream.claim)
        .map_err(|error| {
            CrystalEvidenceBindingError::UpstreamAssessmentRejected(error.to_string())
        })?;

    let expected_source_ids: BTreeSet<String> = upstream
        .lineage
        .assessment
        .source_claim_ids
        .iter()
        .cloned()
        .collect();
    let expected_transition_claim_ids: BTreeSet<String> = upstream
        .lineage
        .assessment
        .transition_evidence_claim_ids
        .iter()
        .cloned()
        .collect();
    let known_transition_ids: BTreeSet<String> = upstream
        .lineage
        .assessment
        .transition_ids
        .iter()
        .cloned()
        .collect();

    let mut actual_source_ids = BTreeSet::new();
    let mut actual_transition_claim_ids = BTreeSet::new();
    let mut all_dependency_ids = BTreeSet::new();

    if upstream.lineage.dependencies.is_empty() {
        return Err(CrystalEvidenceBindingError::UpstreamLineageMismatch);
    }

    let mut recomputed_ceiling = upstream.lineage.dependencies[0].claim.epistemic;
    for dependency in &upstream.lineage.dependencies {
        let claim = &dependency.claim;
        if claim.epistemic.context != EpistemicContext::Scientific {
            return Err(CrystalEvidenceBindingError::UpstreamDependencyNonScientific(
                claim.claim_id.clone(),
            ));
        }
        if !all_dependency_ids.insert(claim.claim_id.clone()) {
            return Err(CrystalEvidenceBindingError::UpstreamLineageMismatch);
        }

        match &dependency.role {
            CrossScaleDependencyRole::Source => {
                actual_source_ids.insert(claim.claim_id.clone());
            }
            CrossScaleDependencyRole::TransitionEvidence { transition_id } => {
                if !known_transition_ids.contains(transition_id) {
                    return Err(CrystalEvidenceBindingError::UnknownTransitionRole(
                        transition_id.clone(),
                    ));
                }
                actual_transition_claim_ids.insert(claim.claim_id.clone());
            }
        }

        recomputed_ceiling.empirical = min(recomputed_ceiling.empirical, claim.epistemic.empirical);
        recomputed_ceiling.normative = min(recomputed_ceiling.normative, claim.epistemic.normative);
        recomputed_ceiling.materiality =
            min(recomputed_ceiling.materiality, claim.epistemic.materiality);
    }

    if actual_source_ids != expected_source_ids
        || actual_transition_claim_ids != expected_transition_claim_ids
        || all_dependency_ids.len() != upstream.lineage.dependencies.len()
    {
        return Err(CrystalEvidenceBindingError::UpstreamLineageMismatch);
    }

    if recomputed_ceiling != upstream.lineage.assessment.authority_ceiling.coordinate {
        return Err(CrystalEvidenceBindingError::UpstreamAuthorityCeilingMismatch);
    }

    let dependency_ood = upstream.lineage.dependencies.iter().any(|dependency| {
        dependency
            .claim
            .uncertainty
            .as_ref()
            .is_some_and(|uncertainty| uncertainty.out_of_domain)
    });
    if dependency_ood
        && !upstream
            .claim
            .uncertainty
            .as_ref()
            .is_some_and(|uncertainty| uncertainty.out_of_domain)
    {
        return Err(CrystalEvidenceBindingError::UpstreamOodStatusNotPropagated);
    }

    Ok(())
}

fn validate_claim_identity(claim: &MatterClaim) -> Result<(), CrystalEvidenceBindingError> {
    if claim.claim_id.trim().is_empty() {
        return Err(CrystalEvidenceBindingError::EmptyEvidenceClaimId);
    }
    if !matches!(claim.scale, MatterScale::Crystal | MatterScale::Multiscale) {
        return Err(CrystalEvidenceBindingError::EvidenceScaleMismatch {
            claim_id: claim.claim_id.clone(),
            actual: claim.scale,
        });
    }
    if claim.epistemic.context != EpistemicContext::Scientific {
        return Err(CrystalEvidenceBindingError::NonScientificEvidence {
            claim_id: claim.claim_id.clone(),
        });
    }
    Ok(())
}

fn validate_required_physics_claim(
    claim: &MatterClaim,
) -> Result<(), CrystalEvidenceBindingError> {
    if !matches!(
        claim.validation_stage,
        MatterValidationStage::PhysicsSimulated
            | MatterValidationStage::CrossModelSupported
            | MatterValidationStage::HighFidelitySimulated
            | MatterValidationStage::ReferenceBenchmarked
    ) {
        return Err(CrystalEvidenceBindingError::InvalidEvidenceStage {
            claim_id: claim.claim_id.clone(),
            stage: claim.validation_stage,
        });
    }

    let solver = claim
        .solver
        .as_ref()
        .ok_or_else(|| CrystalEvidenceBindingError::MissingSolverProfile {
            claim_id: claim.claim_id.clone(),
        })?;
    for (field, value) in [
        ("name", solver.name.as_str()),
        ("version", solver.version.as_str()),
        ("method", solver.method.as_str()),
    ] {
        if value.trim().is_empty() {
            return Err(CrystalEvidenceBindingError::EmptySolverField {
                claim_id: claim.claim_id.clone(),
                field,
            });
        }
    }
    Ok(())
}

fn bind_preregistration(
    slot: CrystalPhaseEvidenceSlot,
    criterion_evidence_id: &str,
    bundle: &ExternalEvidenceBundle,
) -> Result<CrystalCriterionReferenceBinding, CrystalEvidenceBindingError> {
    if criterion_evidence_id.trim().is_empty() {
        return Err(CrystalEvidenceBindingError::EmptyCriterionEvidenceId(slot));
    }
    let reference = bundle.require_role(criterion_evidence_id, EvidenceRole::Preregistration)?;
    Ok(CrystalCriterionReferenceBinding {
        slot,
        evidence_id: reference.id.as_str().to_string(),
        content_sha256: reference.content_sha256.clone(),
        claimed_date: reference.claimed_utc_date,
        reference_interpretation: EvidenceReferenceInterpretation::ReferenceOnly,
    })
}

fn revalidate_relativistic_shape(
    admitted: &AdmittedCrystalPhaseMatterClaim,
    periodic_claim: Option<&MatterClaim>,
    bundle: &ExternalEvidenceBundle,
) -> Result<(), CrystalEvidenceBindingError> {
    match admitted.target.relativistic_requirement {
        Some(requirement) => {
            let relativistic = admitted
                .relativistic_claim
                .as_ref()
                .ok_or(CrystalEvidenceBindingError::MissingRelativisticCrystalEvidence)?;
            let stored_binding = admitted.relativistic_reference_binding.as_ref().ok_or(
                CrystalEvidenceBindingError::MissingRelativisticReferenceBinding,
            )?;

            relativistic.receipt.validate().map_err(|error| {
                CrystalEvidenceBindingError::InvalidRelativisticReceipt(error.to_string())
            })?;
            if relativistic.claim.scale != MatterScale::Crystal {
                return Err(CrystalEvidenceBindingError::RelativisticClaimIsNotCrystal);
            }
            if relativistic.receipt.requirement != requirement {
                return Err(CrystalEvidenceBindingError::RelativisticRequirementMismatch);
            }
            if relativistic.claim.solver.as_ref() != Some(&relativistic.receipt.solver) {
                return Err(CrystalEvidenceBindingError::RelativisticClaimSolverMismatch);
            }
            if !relativistic
                .claim
                .evidence
                .iter()
                .any(|evidence| evidence.evidence_id == relativistic.receipt.receipt_id)
            {
                return Err(CrystalEvidenceBindingError::RelativisticClaimMissingReceiptEvidence);
            }

            let expected_elements: BTreeSet<u16> =
                admitted.target.element_atomic_numbers.iter().copied().collect();
            let actual_elements: BTreeSet<u16> = relativistic
                .receipt
                .element_atomic_numbers
                .iter()
                .copied()
                .collect();
            if expected_elements != actual_elements {
                return Err(CrystalEvidenceBindingError::RelativisticElementSetMismatch);
            }
            if !relativistic
                .receipt
                .solver
                .supports(MatterSolverCapability::PeriodicElectronicStructure)
            {
                return Err(
                    CrystalEvidenceBindingError::RelativisticCrystalLacksPeriodicCapability,
                );
            }
            if periodic_claim != Some(&relativistic.claim) {
                return Err(
                    CrystalEvidenceBindingError::PeriodicEvidenceMustBeRelativisticCrystalClaim,
                );
            }

            let rebound = bind_relativistic_execution_references(&relativistic.receipt, bundle)
                .map_err(|error| {
                    CrystalEvidenceBindingError::RelativisticEvidence(error.to_string())
                })?;
            if &rebound != stored_binding {
                return Err(CrystalEvidenceBindingError::RelativisticBindingMismatch);
            }
        }
        None => {
            if admitted.relativistic_claim.is_some()
                || admitted.relativistic_reference_binding.is_some()
            {
                return Err(CrystalEvidenceBindingError::UnexpectedRelativisticEvidence);
            }
        }
    }
    Ok(())
}

fn require_finite_nonnegative(
    value: f64,
    label: &'static str,
) -> Result<(), CrystalEvidenceBindingError> {
    if !value.is_finite() || value < 0.0 {
        Err(CrystalEvidenceBindingError::InvalidMetric(label))
    } else {
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CrystalEvidenceBindingError {
    ExternalEvidence(ExternalEvidenceError),
    EmptyTargetElementSet,
    InvalidTargetAtomicNumber,
    DuplicateTargetAtomicNumber,
    UpstreamClaimIsNotCrystal,
    UpstreamAssessmentRefused,
    UpstreamAssessmentRejected(String),
    UpstreamLineageMismatch,
    UpstreamDependencyNonScientific(String),
    UnknownTransitionRole(String),
    UpstreamAuthorityCeilingMismatch,
    UpstreamOodStatusNotPropagated,
    EmptyEvidenceClaimId,
    DuplicateEvidenceClaimId(String),
    EvidenceScaleMismatch {
        claim_id: String,
        actual: MatterScale,
    },
    NonScientificEvidence {
        claim_id: String,
    },
    InvalidEvidenceStage {
        claim_id: String,
        stage: MatterValidationStage,
    },
    MissingSolverProfile {
        claim_id: String,
    },
    EmptySolverField {
        claim_id: String,
        field: &'static str,
    },
    DuplicateRequiredEvidenceSlot(CrystalPhaseEvidenceSlot),
    MissingRequiredEvidenceSlot(CrystalPhaseEvidenceSlot),
    SlotDiagnosticMismatch(CrystalPhaseEvidenceSlot),
    PeriodicClaimLacksPeriodicCapability {
        claim_id: String,
    },
    EmptyCriterionEvidenceId(CrystalPhaseEvidenceSlot),
    InvalidMetric(&'static str),
    StoredCriterionNotMet(CrystalPhaseEvidenceSlot),
    NoCompetingPhasesEvaluated,
    NoPhononQPointsEvaluated,
    IncompleteCriterionBindingSet,
    CrystalOodStatusNotPropagated,
    MissingRelativisticCrystalEvidence,
    MissingRelativisticReferenceBinding,
    InvalidRelativisticReceipt(String),
    RelativisticClaimIsNotCrystal,
    RelativisticRequirementMismatch,
    RelativisticClaimSolverMismatch,
    RelativisticClaimMissingReceiptEvidence,
    RelativisticElementSetMismatch,
    RelativisticCrystalLacksPeriodicCapability,
    PeriodicEvidenceMustBeRelativisticCrystalClaim,
    RelativisticEvidence(String),
    RelativisticBindingMismatch,
    UnexpectedRelativisticEvidence,
}

impl From<ExternalEvidenceError> for CrystalEvidenceBindingError {
    fn from(value: ExternalEvidenceError) -> Self {
        Self::ExternalEvidence(value)
    }
}

impl fmt::Display for CrystalEvidenceBindingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ExternalEvidence(error) => write!(f, "{error}"),
            Self::EmptyTargetElementSet => write!(f, "crystal target element set is empty"),
            Self::InvalidTargetAtomicNumber => {
                write!(f, "crystal target contains atomic number zero")
            }
            Self::DuplicateTargetAtomicNumber => {
                write!(f, "crystal target contains duplicate atomic numbers")
            }
            Self::UpstreamClaimIsNotCrystal => {
                write!(f, "upstream admitted claim is not Crystal-scale")
            }
            Self::UpstreamAssessmentRefused => {
                write!(f, "stored upstream cross-scale assessment is refused")
            }
            Self::UpstreamAssessmentRejected(error) => write!(
                f,
                "upstream cross-scale assessment rejects the stored derived claim: {error}"
            ),
            Self::UpstreamLineageMismatch => write!(
                f,
                "upstream lineage snapshots do not exactly match assessment source/transition claim IDs"
            ),
            Self::UpstreamDependencyNonScientific(id) => write!(
                f,
                "upstream dependency claim `{id}` is not in Scientific epistemic context"
            ),
            Self::UnknownTransitionRole(id) => write!(
                f,
                "upstream transition-evidence role references unknown transition `{id}`"
            ),
            Self::UpstreamAuthorityCeilingMismatch => write!(
                f,
                "stored upstream authority ceiling does not equal the componentwise minimum of its dependency snapshots"
            ),
            Self::UpstreamOodStatusNotPropagated => write!(
                f,
                "OOD upstream dependency evidence is not propagated to the stored derived crystal claim"
            ),
            Self::EmptyEvidenceClaimId => write!(f, "crystal evidence claim id is empty"),
            Self::DuplicateEvidenceClaimId(id) => {
                write!(f, "crystal evidence claim `{id}` appears more than once")
            }
            Self::EvidenceScaleMismatch { claim_id, actual } => write!(
                f,
                "crystal evidence claim `{claim_id}` addresses {actual:?}, expected Crystal or Multiscale"
            ),
            Self::NonScientificEvidence { claim_id } => write!(
                f,
                "crystal evidence claim `{claim_id}` is not in Scientific epistemic context"
            ),
            Self::InvalidEvidenceStage { claim_id, stage } => write!(
                f,
                "required crystal evidence claim `{claim_id}` has inadmissible stage {stage:?}"
            ),
            Self::MissingSolverProfile { claim_id } => write!(
                f,
                "required crystal evidence claim `{claim_id}` has no solver provenance"
            ),
            Self::EmptySolverField { claim_id, field } => write!(
                f,
                "required crystal evidence claim `{claim_id}` has empty solver {field}"
            ),
            Self::DuplicateRequiredEvidenceSlot(slot) => {
                write!(f, "required crystal evidence slot {slot:?} appears more than once")
            }
            Self::MissingRequiredEvidenceSlot(slot) => {
                write!(f, "required crystal evidence slot {slot:?} is missing")
            }
            Self::SlotDiagnosticMismatch(slot) => write!(
                f,
                "crystal evidence slot {slot:?} does not match its stored diagnostic variant"
            ),
            Self::PeriodicClaimLacksPeriodicCapability { claim_id } => write!(
                f,
                "periodic crystal evidence claim `{claim_id}` does not advertise PeriodicElectronicStructure"
            ),
            Self::EmptyCriterionEvidenceId(slot) => write!(
                f,
                "crystal criterion evidence id for {slot:?} must not be empty"
            ),
            Self::InvalidMetric(label) => write!(f, "stored crystal metric `{label}` is invalid"),
            Self::StoredCriterionNotMet(slot) => write!(
                f,
                "stored crystal diagnostic for {slot:?} does not meet its declared criterion"
            ),
            Self::NoCompetingPhasesEvaluated => {
                write!(f, "stored phase-competition evidence contains zero competing phases")
            }
            Self::NoPhononQPointsEvaluated => {
                write!(f, "stored dynamical-stability evidence contains zero q-points")
            }
            Self::IncompleteCriterionBindingSet => write!(
                f,
                "exactly relaxation, phase-competition, and dynamical criteria must bind to preregistration references"
            ),
            Self::CrystalOodStatusNotPropagated => write!(
                f,
                "OOD required crystal evidence is not propagated to the upstream crystal claim"
            ),
            Self::MissingRelativisticCrystalEvidence => write!(
                f,
                "relativistic crystal target is missing its admitted relativistic claim"
            ),
            Self::MissingRelativisticReferenceBinding => write!(
                f,
                "relativistic crystal target is missing its canonical execution-reference binding"
            ),
            Self::InvalidRelativisticReceipt(error) => {
                write!(f, "relativistic crystal receipt is invalid: {error}")
            }
            Self::RelativisticClaimIsNotCrystal => {
                write!(f, "relativistic crystal evidence must itself be Crystal-scale")
            }
            Self::RelativisticRequirementMismatch => {
                write!(f, "relativistic crystal requirement does not match the target")
            }
            Self::RelativisticClaimSolverMismatch => write!(
                f,
                "stored relativistic Matter claim solver profile differs from its execution receipt"
            ),
            Self::RelativisticClaimMissingReceiptEvidence => write!(
                f,
                "stored relativistic Matter claim does not retain its execution receipt evidence id"
            ),
            Self::RelativisticElementSetMismatch => {
                write!(f, "relativistic crystal element set does not match the target")
            }
            Self::RelativisticCrystalLacksPeriodicCapability => write!(
                f,
                "relativistic crystal solver does not advertise PeriodicElectronicStructure"
            ),
            Self::PeriodicEvidenceMustBeRelativisticCrystalClaim => write!(
                f,
                "periodic evidence must exactly equal the admitted relativistic Crystal claim"
            ),
            Self::RelativisticEvidence(error) => {
                write!(f, "relativistic execution evidence revalidation failed: {error}")
            }
            Self::RelativisticBindingMismatch => write!(
                f,
                "stored relativistic execution binding differs from a fresh canonical rebind"
            ),
            Self::UnexpectedRelativisticEvidence => write!(
                f,
                "non-relativistic crystal target unexpectedly carries relativistic admission fields"
            ),
        }
    }
}

impl std::error::Error for CrystalEvidenceBindingError {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::matter_cross_scale::MatterScaleTransition;
    use crate::knowledge::matter_cross_scale_lineage::CrossScaleLineageEnvelope;
    use crate::knowledge::matter_crystal_phase::{
        admit_crystal_phase_candidate, CrystalPhaseEvidence, CrystalPhaseTarget,
    };
    use symthaea_epistemic_types::{
        EmpiricalLevel, EpistemicCoordinate, MaterialityLevel, MatterSolverProfile,
        NormativeLevel,
    };
    use symthaea_evidence_plane::external_receipt::ExternalEvidenceReference;

    fn digest(ch: char) -> String {
        std::iter::repeat_n(ch, 64).collect()
    }

    fn reference(
        id: &str,
        role: EvidenceRole,
        digest_char: char,
    ) -> ExternalEvidenceReference {
        ExternalEvidenceReference::try_new(
            id,
            role,
            digest(digest_char),
            format!("artifact://{id}"),
            format!("subject:{id}"),
            Some(20260101),
            Some("fixture issuer".to_string()),
        )
        .unwrap()
    }

    fn bundle() -> ExternalEvidenceBundle {
        ExternalEvidenceBundle::new(vec![
            reference("criterion:relax:v1", EvidenceRole::Preregistration, 'a'),
            reference("criterion:hull:v1", EvidenceRole::Preregistration, 'b'),
            reference("criterion:phonon:v1", EvidenceRole::Preregistration, 'c'),
        ])
        .unwrap()
    }

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
            id,
            scale,
            MatterValidationStage::HighFidelitySimulated,
            solver(id, capabilities),
        )
        .unwrap()
    }

    fn admitted() -> AdmittedCrystalPhaseMatterClaim {
        let source = claim("source", MatterScale::AtomicElectronic, vec![]);
        let transition_claim = claim("transition", MatterScale::Crystal, vec![]);
        let transition = MatterScaleTransition::supported_in_domain(
            "electronic->crystal",
            MatterScale::AtomicElectronic,
            MatterScale::Crystal,
            &transition_claim,
        );
        let lineage = CrossScaleLineageEnvelope::assess(
            &[&source],
            &[transition],
            MatterScale::Crystal,
        )
        .unwrap();
        let upstream = lineage
            .admit_derived_computational_claim(claim("candidate", MatterScale::Crystal, vec![]))
            .unwrap();

        let periodic = claim(
            "periodic",
            MatterScale::Crystal,
            vec![MatterSolverCapability::PeriodicElectronicStructure],
        );
        let relaxation = claim("relax", MatterScale::Crystal, vec![]);
        let phase = claim("phase", MatterScale::Crystal, vec![]);
        let dynamic = claim("dynamic", MatterScale::Crystal, vec![]);
        let evidence = vec![
            CrystalPhaseEvidence::periodic_electronic_structure(&periodic).unwrap(),
            CrystalPhaseEvidence::structural_relaxation(
                &relaxation,
                0.01,
                0.02,
                "criterion:relax:v1",
            )
            .unwrap(),
            CrystalPhaseEvidence::thermodynamic_phase_competition(
                &phase,
                0.01,
                0.05,
                8,
                "criterion:hull:v1",
            )
            .unwrap(),
            CrystalPhaseEvidence::dynamical_stability(
                &dynamic,
                -0.01,
                0.02,
                32,
                "criterion:phonon:v1",
            )
            .unwrap(),
        ];
        admit_crystal_phase_candidate(
            upstream,
            CrystalPhaseTarget::try_new(vec![14, 8], None).unwrap(),
            &evidence,
            None,
            None,
        )
        .unwrap()
    }

    #[test]
    fn intact_admission_binds_all_preregistered_criteria() {
        let bound = bind_crystal_phase_evidence(admitted(), &bundle()).unwrap();
        assert_eq!(bound.criterion_bindings.len(), 3);
        assert!(bound.criterion_bindings.iter().all(|binding| {
            binding.reference_interpretation == EvidenceReferenceInterpretation::ReferenceOnly
        }));
        assert_eq!(
            bound
                .criterion_bindings
                .iter()
                .find(|binding| binding.slot == CrystalPhaseEvidenceSlot::StructuralRelaxation)
                .unwrap()
                .content_sha256
                .as_str(),
            digest('a')
        );
    }

    #[test]
    fn wrong_preregistration_role_fails_closed() {
        let bad = ExternalEvidenceBundle::new(vec![
            reference("criterion:relax:v1", EvidenceRole::ArtifactContent, 'a'),
            reference("criterion:hull:v1", EvidenceRole::Preregistration, 'b'),
            reference("criterion:phonon:v1", EvidenceRole::Preregistration, 'c'),
        ])
        .unwrap();
        assert!(matches!(
            bind_crystal_phase_evidence(admitted(), &bad),
            Err(CrystalEvidenceBindingError::ExternalEvidence(
                ExternalEvidenceError::RoleMismatch { .. }
            ))
        ));
    }

    #[test]
    fn missing_solver_provenance_is_rejected() {
        let mut admitted = admitted();
        let relaxation = admitted
            .crystal_evidence
            .iter_mut()
            .find(|snapshot| snapshot.slot == CrystalPhaseEvidenceSlot::StructuralRelaxation)
            .unwrap();
        relaxation.claim.solver = None;
        assert!(matches!(
            bind_crystal_phase_evidence(admitted, &bundle()),
            Err(CrystalEvidenceBindingError::MissingSolverProfile { .. })
        ));
    }

    #[test]
    fn tampered_stored_relaxation_metric_is_rejected() {
        let mut admitted = admitted();
        let relaxation = admitted
            .crystal_evidence
            .iter_mut()
            .find(|snapshot| snapshot.slot == CrystalPhaseEvidenceSlot::StructuralRelaxation)
            .unwrap();
        relaxation.diagnostic = CrystalPhaseDiagnosticSnapshot::StructuralRelaxation {
            max_force_bits: 0.5_f64.to_bits(),
            force_tolerance_bits: 0.02_f64.to_bits(),
            criterion_evidence_id: "criterion:relax:v1".to_string(),
        };
        assert_eq!(
            bind_crystal_phase_evidence(admitted, &bundle()).unwrap_err(),
            CrystalEvidenceBindingError::StoredCriterionNotMet(
                CrystalPhaseEvidenceSlot::StructuralRelaxation
            )
        );
    }

    #[test]
    fn slot_diagnostic_mismatch_is_rejected() {
        let mut admitted = admitted();
        let phase = admitted
            .crystal_evidence
            .iter_mut()
            .find(|snapshot| {
                snapshot.slot == CrystalPhaseEvidenceSlot::ThermodynamicPhaseCompetition
            })
            .unwrap();
        phase.diagnostic = CrystalPhaseDiagnosticSnapshot::HeuristicCompositionScreening;
        assert_eq!(
            bind_crystal_phase_evidence(admitted, &bundle()).unwrap_err(),
            CrystalEvidenceBindingError::SlotDiagnosticMismatch(
                CrystalPhaseEvidenceSlot::ThermodynamicPhaseCompetition
            )
        );
    }

    #[test]
    fn tampered_target_identity_fails_before_binding() {
        let mut admitted = admitted();
        admitted.target.element_atomic_numbers = vec![14, 14];
        assert_eq!(
            bind_crystal_phase_evidence(admitted, &bundle()).unwrap_err(),
            CrystalEvidenceBindingError::DuplicateTargetAtomicNumber
        );
    }

    #[test]
    fn tampered_non_scientific_evidence_fails_before_binding() {
        let mut admitted = admitted();
        let snapshot = admitted
            .crystal_evidence
            .iter_mut()
            .find(|snapshot| snapshot.slot == CrystalPhaseEvidenceSlot::StructuralRelaxation)
            .unwrap();
        snapshot.claim.epistemic.context = EpistemicContext::Personal;
        assert!(matches!(
            bind_crystal_phase_evidence(admitted, &bundle()),
            Err(CrystalEvidenceBindingError::NonScientificEvidence { .. })
        ));
    }

    #[test]
    fn tampered_surrogate_required_slot_fails_before_binding() {
        let mut admitted = admitted();
        let snapshot = admitted
            .crystal_evidence
            .iter_mut()
            .find(|snapshot| snapshot.slot == CrystalPhaseEvidenceSlot::DynamicalStability)
            .unwrap();
        snapshot.claim.validation_stage = MatterValidationStage::SurrogateSupported;
        assert!(matches!(
            bind_crystal_phase_evidence(admitted, &bundle()),
            Err(CrystalEvidenceBindingError::InvalidEvidenceStage { .. })
        ));
    }

    #[test]
    fn tampered_upstream_authority_is_rejected() {
        let mut admitted = admitted();
        admitted.upstream.claim.epistemic = EpistemicCoordinate::new(
            EmpiricalLevel::E4PubliclyReproducible,
            NormativeLevel::N2Network,
            MaterialityLevel::M2Persistent,
            EpistemicContext::Scientific,
        );
        assert!(matches!(
            bind_crystal_phase_evidence(admitted, &bundle()),
            Err(CrystalEvidenceBindingError::UpstreamAssessmentRejected(_))
        ));
    }

    #[test]
    fn tampered_upstream_dependency_set_is_rejected() {
        let mut admitted = admitted();
        admitted.upstream.lineage.dependencies.pop();
        assert_eq!(
            bind_crystal_phase_evidence(admitted, &bundle()).unwrap_err(),
            CrystalEvidenceBindingError::UpstreamLineageMismatch
        );
    }

    #[test]
    fn tampered_upstream_authority_ceiling_is_rejected() {
        let mut admitted = admitted();
        admitted.upstream.lineage.assessment.authority_ceiling.coordinate.empirical =
            EmpiricalLevel::E4PubliclyReproducible;
        assert_eq!(
            bind_crystal_phase_evidence(admitted, &bundle()).unwrap_err(),
            CrystalEvidenceBindingError::UpstreamAuthorityCeilingMismatch
        );
    }
}
