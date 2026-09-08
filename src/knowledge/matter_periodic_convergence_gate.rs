// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Numerical convergence boundary for periodic crystal energies.
//!
//! Every unique periodic energy used by the crystal execution plan must carry a
//! two-stage convergence study: plane-wave cutoff convergence followed by
//! k-point-grid convergence. The structure, fixed physics settings, solver,
//! parser/environment, and frozen dependency set remain invariant while the
//! numerical resolution is increased.
//!
//! Passing this gate means only that the final two deltas on each declared axis
//! lie inside a preregistered tolerance. It is not a proof of complete-basis,
//! Brillouin-zone, DFT-functional, pseudopotential, or experimental accuracy.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use symthaea_epistemic_types::{MatterClaim, MatterSolverProfile};
use symthaea_evidence_plane::external_receipt::{
    DeclaredChronologyInterpretation, DeclaredTemporalRelation, EvidenceReferenceInterpretation,
    EvidenceRole, ExternalEvidenceBundle, ExternalEvidenceError, Sha256Digest,
};

use super::matter_crystal_capability_gate::bind_capability_gated_crystal_execution_plan;
use super::matter_crystal_evidence_binding::EvidenceBoundCrystalPhaseMatterClaim;
use super::matter_crystal_execution::{
    CrystalExecutionError, CrystalSolverExecutionBinding, CrystalSolverExecutionReceipt,
    CrystalSolverTask,
};
use super::matter_crystal_leaf_binding::bind_crystal_solver_leaf;

pub use super::matter_crystal_capability_gate::{
    CapabilityBoundCrystalExecutionPlan, CrystalCapabilityGateError,
    CrystalCapabilityInterpretation, CrystalClaimCapabilityBinding, CrystalClaimCapabilityReceipt,
    CrystalScientificCapability, MixedCrystalExecutionPlanReceipt,
};

const MIN_AXIS_SAMPLES: usize = 3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PeriodicConvergenceInterpretation {
    ObservedFinalTwoDeltasWithinDeclaredTolerance,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PeriodicEnergyConvergenceSampleReceipt {
    pub sample_id: String,
    pub claim: MatterClaim,
    pub execution: CrystalSolverExecutionReceipt,
    pub energy_output_artifact_id: String,
    pub energy_per_atom_bits: u64,
    pub wavefunction_cutoff_ev_bits: u64,
    pub charge_density_cutoff_ev_bits: u64,
    pub k_grid: [u32; 3],
    pub k_shift_fractional_bits: [u64; 3],
    pub normalized_structure_artifact_id: String,
    pub fixed_settings_artifact_id: String,
    pub pseudopotential_snapshot_id: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PeriodicEnergyConvergenceStudyReceipt {
    pub production_claim_id: String,
    pub production_execution_evidence_id: String,
    pub criterion_evidence_id: String,
    pub energy_tolerance_ev_per_atom_bits: u64,
    pub cutoff_series: Vec<PeriodicEnergyConvergenceSampleReceipt>,
    pub kpoint_series: Vec<PeriodicEnergyConvergenceSampleReceipt>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PeriodicEnergyConvergenceSampleBinding {
    pub sample_id: String,
    pub execution: CrystalSolverExecutionBinding,
    pub energy_output_evidence_id: String,
    pub energy_output_sha256: Sha256Digest,
    pub energy_per_atom_bits: u64,
    pub wavefunction_cutoff_ev_bits: u64,
    pub charge_density_cutoff_ev_bits: u64,
    pub k_grid: [u32; 3],
    pub k_shift_fractional_bits: [u64; 3],
    pub normalized_structure_evidence_id: String,
    pub normalized_structure_sha256: Sha256Digest,
    pub fixed_settings_evidence_id: String,
    pub fixed_settings_sha256: Sha256Digest,
    pub pseudopotential_snapshot_evidence_id: String,
    pub pseudopotential_snapshot_sha256: Sha256Digest,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PeriodicEnergyConvergenceStudyBinding {
    pub production_claim_id: String,
    pub production_execution_evidence_id: String,
    pub criterion_evidence_id: String,
    pub criterion_sha256: Sha256Digest,
    pub energy_tolerance_ev_per_atom_bits: u64,
    pub cutoff_series: Vec<PeriodicEnergyConvergenceSampleBinding>,
    pub kpoint_series: Vec<PeriodicEnergyConvergenceSampleBinding>,
    pub criterion_chronology: Vec<DeclaredTemporalRelation>,
    pub reference_interpretation: EvidenceReferenceInterpretation,
    pub chronology_interpretation: DeclaredChronologyInterpretation,
    pub convergence_interpretation: PeriodicConvergenceInterpretation,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ConvergenceBoundCrystalExecutionPlan {
    pub capability_bound: CapabilityBoundCrystalExecutionPlan,
    pub periodic_energy_studies: Vec<PeriodicEnergyConvergenceStudyBinding>,
    pub reference_interpretation: EvidenceReferenceInterpretation,
    pub convergence_interpretation: PeriodicConvergenceInterpretation,
}

#[derive(Debug, Clone)]
struct ProductionPeriodicTarget {
    claim: MatterClaim,
    execution: CrystalSolverExecutionBinding,
}

pub fn bind_converged_crystal_execution_plan(
    evidence_bound: EvidenceBoundCrystalPhaseMatterClaim,
    plan_receipt: MixedCrystalExecutionPlanReceipt,
    capability_receipts: &[CrystalClaimCapabilityReceipt],
    studies: &[PeriodicEnergyConvergenceStudyReceipt],
    bundle: &ExternalEvidenceBundle,
) -> Result<ConvergenceBoundCrystalExecutionPlan, PeriodicConvergenceError> {
    let capability_bound = bind_capability_gated_crystal_execution_plan(
        evidence_bound,
        plan_receipt,
        capability_receipts,
        bundle,
    )?;
    let targets = collect_unique_periodic_targets(&capability_bound)?;

    let mut supplied = BTreeMap::new();
    for study in studies {
        let key = study.production_execution_evidence_id.clone();
        if key.trim().is_empty() {
            return Err(PeriodicConvergenceError::EmptyProductionExecutionId);
        }
        if supplied.insert(key.clone(), study).is_some() {
            return Err(PeriodicConvergenceError::DuplicateStudyForExecution(key));
        }
    }
    let expected: BTreeSet<_> = targets.keys().cloned().collect();
    let actual: BTreeSet<_> = supplied.keys().cloned().collect();
    if expected != actual {
        return Err(PeriodicConvergenceError::StudySetMismatch);
    }

    let mut bound_studies = Vec::with_capacity(targets.len());
    for (execution_id, target) in targets {
        let study = supplied
            .get(&execution_id)
            .expect("periodic convergence study key set checked above");
        bound_studies.push(bind_study(target, study, bundle)?);
    }
    bound_studies.sort_by(|a, b| {
        a.production_execution_evidence_id
            .cmp(&b.production_execution_evidence_id)
    });

    Ok(ConvergenceBoundCrystalExecutionPlan {
        capability_bound,
        periodic_energy_studies: bound_studies,
        reference_interpretation: EvidenceReferenceInterpretation::ReferenceOnly,
        convergence_interpretation:
            PeriodicConvergenceInterpretation::ObservedFinalTwoDeltasWithinDeclaredTolerance,
    })
}

fn collect_unique_periodic_targets(
    parent: &CapabilityBoundCrystalExecutionPlan,
) -> Result<BTreeMap<String, ProductionPeriodicTarget>, PeriodicConvergenceError> {
    let plan = &parent.plan;
    let mut targets = BTreeMap::new();
    let mut digest_to_id = BTreeMap::new();

    insert_target(
        &mut targets,
        &mut digest_to_id,
        &plan.periodic_claim,
        &plan.periodic_execution,
    )?;
    for leaf in &plan.phase_competition.leaves {
        insert_target(
            &mut targets,
            &mut digest_to_id,
            &leaf.claim,
            &leaf.execution,
        )?;
    }
    Ok(targets)
}

fn insert_target(
    targets: &mut BTreeMap<String, ProductionPeriodicTarget>,
    digest_to_id: &mut BTreeMap<String, String>,
    claim: &MatterClaim,
    execution: &CrystalSolverExecutionBinding,
) -> Result<(), PeriodicConvergenceError> {
    let id = execution.execution_evidence_id.clone();
    let digest = execution.execution_content_sha256.as_str().to_string();
    if let Some(existing_id) = digest_to_id.insert(digest, id.clone()) {
        if existing_id != id {
            return Err(PeriodicConvergenceError::SameProductionContentDifferentIds);
        }
    }
    match targets.get(&id) {
        Some(existing) => {
            if existing.claim != *claim || existing.execution != *execution {
                return Err(PeriodicConvergenceError::InconsistentProductionReuse(id));
            }
        }
        None => {
            targets.insert(
                id,
                ProductionPeriodicTarget {
                    claim: claim.clone(),
                    execution: execution.clone(),
                },
            );
        }
    }
    Ok(())
}

fn bind_study(
    production: ProductionPeriodicTarget,
    study: &PeriodicEnergyConvergenceStudyReceipt,
    bundle: &ExternalEvidenceBundle,
) -> Result<PeriodicEnergyConvergenceStudyBinding, PeriodicConvergenceError> {
    if study.production_claim_id != production.claim.claim_id
        || study.production_execution_evidence_id != production.execution.execution_evidence_id
    {
        return Err(PeriodicConvergenceError::ProductionIdentityMismatch);
    }
    if study.criterion_evidence_id.trim().is_empty() {
        return Err(PeriodicConvergenceError::EmptyCriterionId);
    }
    let tolerance = f64::from_bits(study.energy_tolerance_ev_per_atom_bits);
    if !tolerance.is_finite() || tolerance <= 0.0 {
        return Err(PeriodicConvergenceError::InvalidEnergyTolerance);
    }
    if study.cutoff_series.len() < MIN_AXIS_SAMPLES || study.kpoint_series.len() < MIN_AXIS_SAMPLES {
        return Err(PeriodicConvergenceError::InsufficientAxisSamples);
    }

    let criterion = bundle.require_role(&study.criterion_evidence_id, EvidenceRole::Preregistration)?;
    if !production
        .claim
        .evidence
        .iter()
        .any(|evidence| evidence.evidence_id == study.criterion_evidence_id)
    {
        return Err(PeriodicConvergenceError::CriterionMissingFromProductionClaim);
    }

    let production_solver = production
        .claim
        .solver
        .as_ref()
        .ok_or(PeriodicConvergenceError::ProductionSolverMissing)?
        .clone();

    let mut execution_ids = BTreeSet::new();
    let mut execution_digests = BTreeSet::new();
    let mut sample_ids = BTreeSet::new();
    let mut chronology = Vec::new();

    let cutoff = bind_series(
        &study.cutoff_series,
        &production_solver,
        &study.criterion_evidence_id,
        bundle,
        &mut execution_ids,
        &mut execution_digests,
        &mut sample_ids,
        &mut chronology,
    )?;
    let kpoint = bind_series(
        &study.kpoint_series,
        &production_solver,
        &study.criterion_evidence_id,
        bundle,
        &mut execution_ids,
        &mut execution_digests,
        &mut sample_ids,
        &mut chronology,
    )?;

    validate_invariant_artifacts(&cutoff, &kpoint)?;
    validate_cutoff_axis(&cutoff, tolerance)?;
    validate_kpoint_axis(&cutoff, &kpoint, tolerance)?;

    let final_kpoint = kpoint
        .last()
        .ok_or(PeriodicConvergenceError::InsufficientAxisSamples)?;
    let final_receipt = study
        .kpoint_series
        .last()
        .ok_or(PeriodicConvergenceError::InsufficientAxisSamples)?;
    if final_receipt.claim != production.claim || final_kpoint.execution != production.execution {
        return Err(PeriodicConvergenceError::ProductionIsNotFinalKPointSample);
    }

    let any_ood = study
        .cutoff_series
        .iter()
        .chain(study.kpoint_series.iter())
        .any(|sample| {
            sample
                .claim
                .uncertainty
                .as_ref()
                .is_some_and(|uncertainty| uncertainty.out_of_domain)
        });
    if any_ood
        && !production
            .claim
            .uncertainty
            .as_ref()
            .is_some_and(|uncertainty| uncertainty.out_of_domain)
    {
        return Err(PeriodicConvergenceError::SampleOodNotPropagated);
    }

    Ok(PeriodicEnergyConvergenceStudyBinding {
        production_claim_id: production.claim.claim_id,
        production_execution_evidence_id: production.execution.execution_evidence_id,
        criterion_evidence_id: criterion.id.as_str().to_string(),
        criterion_sha256: criterion.content_sha256.clone(),
        energy_tolerance_ev_per_atom_bits: study.energy_tolerance_ev_per_atom_bits,
        cutoff_series: cutoff,
        kpoint_series: kpoint,
        criterion_chronology: chronology,
        reference_interpretation: EvidenceReferenceInterpretation::ReferenceOnly,
        chronology_interpretation: DeclaredChronologyInterpretation::DeclaredChronologyOnly,
        convergence_interpretation:
            PeriodicConvergenceInterpretation::ObservedFinalTwoDeltasWithinDeclaredTolerance,
    })
}

#[allow(clippy::too_many_arguments)]
fn bind_series(
    series: &[PeriodicEnergyConvergenceSampleReceipt],
    production_solver: &MatterSolverProfile,
    criterion_id: &str,
    bundle: &ExternalEvidenceBundle,
    execution_ids: &mut BTreeSet<String>,
    execution_digests: &mut BTreeSet<String>,
    sample_ids: &mut BTreeSet<String>,
    chronology: &mut Vec<DeclaredTemporalRelation>,
) -> Result<Vec<PeriodicEnergyConvergenceSampleBinding>, PeriodicConvergenceError> {
    let mut out = Vec::with_capacity(series.len());
    for sample in series {
        validate_sample_shape(sample)?;
        if !sample_ids.insert(sample.sample_id.clone()) {
            return Err(PeriodicConvergenceError::DuplicateSampleId(
                sample.sample_id.clone(),
            ));
        }
        if sample.claim.solver.as_ref() != Some(production_solver) {
            return Err(PeriodicConvergenceError::SolverProfileDrift(
                sample.sample_id.clone(),
            ));
        }
        if !sample
            .claim
            .evidence
            .iter()
            .any(|evidence| evidence.evidence_id == criterion_id)
        {
            return Err(PeriodicConvergenceError::CriterionMissingFromSampleClaim(
                sample.sample_id.clone(),
            ));
        }

        let execution = bind_crystal_solver_leaf(&sample.execution, &sample.claim, bundle)?;
        if !execution_ids.insert(execution.execution_evidence_id.clone()) {
            return Err(PeriodicConvergenceError::DuplicateSampleExecutionId(
                execution.execution_evidence_id.clone(),
            ));
        }
        if !execution_digests.insert(execution.execution_content_sha256.as_str().to_string()) {
            return Err(PeriodicConvergenceError::DuplicateSampleExecutionContent);
        }

        let energy = bundle.require_role(
            &sample.energy_output_artifact_id,
            EvidenceRole::ArtifactContent,
        )?;
        let structure = bundle.require_role(
            &sample.normalized_structure_artifact_id,
            EvidenceRole::ArtifactContent,
        )?;
        let fixed = bundle.require_role(
            &sample.fixed_settings_artifact_id,
            EvidenceRole::ArtifactContent,
        )?;
        let pseudo = bundle.require_role(
            &sample.pseudopotential_snapshot_id,
            EvidenceRole::DependencySnapshot,
        )?;

        for id in [
            sample.energy_output_artifact_id.as_str(),
            sample.normalized_structure_artifact_id.as_str(),
            sample.fixed_settings_artifact_id.as_str(),
            sample.pseudopotential_snapshot_id.as_str(),
        ] {
            if !sample
                .claim
                .evidence
                .iter()
                .any(|evidence| evidence.evidence_id == id)
            {
                return Err(PeriodicConvergenceError::InvariantEvidenceMissingFromClaim {
                    sample_id: sample.sample_id.clone(),
                    evidence_id: id.to_string(),
                });
            }
        }
        if !sample
            .execution
            .output_artifact_ids
            .contains(&sample.energy_output_artifact_id)
        {
            return Err(PeriodicConvergenceError::EnergyNotExecutionOutput(
                sample.sample_id.clone(),
            ));
        }
        if !sample
            .execution
            .dependency_snapshot_ids
            .contains(&sample.pseudopotential_snapshot_id)
        {
            return Err(PeriodicConvergenceError::PseudopotentialNotExecutionDependency(
                sample.sample_id.clone(),
            ));
        }

        let execution_date = execution.execution_claimed_date;
        for reference in [structure, fixed, pseudo] {
            let date = reference.claimed_utc_date.ok_or_else(|| {
                PeriodicConvergenceError::InvariantEvidenceMissingDate(
                    reference.id.as_str().to_string(),
                )
            })?;
            if date > execution_date {
                return Err(PeriodicConvergenceError::InvariantEvidencePostdatesExecution(
                    reference.id.as_str().to_string(),
                ));
            }
        }
        chronology.push(bundle.require_preregistration_before(
            criterion_id,
            &execution.execution_evidence_id,
        )?);

        out.push(PeriodicEnergyConvergenceSampleBinding {
            sample_id: sample.sample_id.clone(),
            execution,
            energy_output_evidence_id: energy.id.as_str().to_string(),
            energy_output_sha256: energy.content_sha256.clone(),
            energy_per_atom_bits: sample.energy_per_atom_bits,
            wavefunction_cutoff_ev_bits: sample.wavefunction_cutoff_ev_bits,
            charge_density_cutoff_ev_bits: sample.charge_density_cutoff_ev_bits,
            k_grid: sample.k_grid,
            k_shift_fractional_bits: sample.k_shift_fractional_bits,
            normalized_structure_evidence_id: structure.id.as_str().to_string(),
            normalized_structure_sha256: structure.content_sha256.clone(),
            fixed_settings_evidence_id: fixed.id.as_str().to_string(),
            fixed_settings_sha256: fixed.content_sha256.clone(),
            pseudopotential_snapshot_evidence_id: pseudo.id.as_str().to_string(),
            pseudopotential_snapshot_sha256: pseudo.content_sha256.clone(),
        });
    }
    Ok(out)
}

fn validate_sample_shape(
    sample: &PeriodicEnergyConvergenceSampleReceipt,
) -> Result<(), PeriodicConvergenceError> {
    if sample.sample_id.trim().is_empty()
        || sample.energy_output_artifact_id.trim().is_empty()
        || sample.normalized_structure_artifact_id.trim().is_empty()
        || sample.fixed_settings_artifact_id.trim().is_empty()
        || sample.pseudopotential_snapshot_id.trim().is_empty()
    {
        return Err(PeriodicConvergenceError::EmptySampleField);
    }
    if sample.execution.task != CrystalSolverTask::PeriodicElectronicStructure {
        return Err(PeriodicConvergenceError::SampleIsNotPeriodicElectronicStructure);
    }
    if !f64::from_bits(sample.energy_per_atom_bits).is_finite() {
        return Err(PeriodicConvergenceError::InvalidEnergy);
    }
    for bits in [
        sample.wavefunction_cutoff_ev_bits,
        sample.charge_density_cutoff_ev_bits,
    ] {
        let value = f64::from_bits(bits);
        if !value.is_finite() || value <= 0.0 {
            return Err(PeriodicConvergenceError::InvalidCutoff);
        }
    }
    if sample.k_grid.iter().any(|value| *value == 0) {
        return Err(PeriodicConvergenceError::InvalidKGrid);
    }
    if sample
        .k_shift_fractional_bits
        .iter()
        .any(|bits| !f64::from_bits(*bits).is_finite())
    {
        return Err(PeriodicConvergenceError::InvalidKShift);
    }
    Ok(())
}

fn validate_invariant_artifacts(
    cutoff: &[PeriodicEnergyConvergenceSampleBinding],
    kpoint: &[PeriodicEnergyConvergenceSampleBinding],
) -> Result<(), PeriodicConvergenceError> {
    let first = cutoff
        .first()
        .ok_or(PeriodicConvergenceError::InsufficientAxisSamples)?;
    for sample in cutoff.iter().chain(kpoint.iter()) {
        if sample.normalized_structure_sha256 != first.normalized_structure_sha256
            || sample.fixed_settings_sha256 != first.fixed_settings_sha256
            || sample.pseudopotential_snapshot_sha256 != first.pseudopotential_snapshot_sha256
            || sample.execution.environment_content_sha256
                != first.execution.environment_content_sha256
            || sample.execution.parser_content_sha256 != first.execution.parser_content_sha256
            || dependency_set(&sample.execution) != dependency_set(&first.execution)
            || sample.execution.adapter_name != first.execution.adapter_name
            || sample.execution.adapter_version != first.execution.adapter_version
        {
            return Err(PeriodicConvergenceError::NonResolutionSettingsDrift);
        }
    }
    Ok(())
}

fn dependency_set(binding: &CrystalSolverExecutionBinding) -> BTreeSet<String> {
    binding
        .dependency_content_sha256
        .iter()
        .map(|digest| digest.as_str().to_string())
        .collect()
}

fn validate_cutoff_axis(
    series: &[PeriodicEnergyConvergenceSampleBinding],
    tolerance: f64,
) -> Result<(), PeriodicConvergenceError> {
    let first = &series[0];
    for sample in series {
        if sample.k_grid != first.k_grid
            || sample.k_shift_fractional_bits != first.k_shift_fractional_bits
        {
            return Err(PeriodicConvergenceError::CutoffAxisKPointDrift);
        }
    }
    for pair in series.windows(2) {
        let previous_wfc = f64::from_bits(pair[0].wavefunction_cutoff_ev_bits);
        let current_wfc = f64::from_bits(pair[1].wavefunction_cutoff_ev_bits);
        let previous_rho = f64::from_bits(pair[0].charge_density_cutoff_ev_bits);
        let current_rho = f64::from_bits(pair[1].charge_density_cutoff_ev_bits);
        if current_wfc < previous_wfc
            || current_rho < previous_rho
            || (current_wfc == previous_wfc && current_rho == previous_rho)
        {
            return Err(PeriodicConvergenceError::CutoffAxisNotIncreasing);
        }
    }
    require_final_two_deltas(series, tolerance)
}

fn validate_kpoint_axis(
    cutoff: &[PeriodicEnergyConvergenceSampleBinding],
    series: &[PeriodicEnergyConvergenceSampleBinding],
    tolerance: f64,
) -> Result<(), PeriodicConvergenceError> {
    let cutoff_final = cutoff
        .last()
        .ok_or(PeriodicConvergenceError::InsufficientAxisSamples)?;
    let first = &series[0];
    for sample in series {
        if sample.wavefunction_cutoff_ev_bits != cutoff_final.wavefunction_cutoff_ev_bits
            || sample.charge_density_cutoff_ev_bits != cutoff_final.charge_density_cutoff_ev_bits
            || sample.k_shift_fractional_bits != first.k_shift_fractional_bits
        {
            return Err(PeriodicConvergenceError::KPointAxisNonGridDrift);
        }
    }
    if first.k_grid != cutoff_final.k_grid
        || first.k_shift_fractional_bits != cutoff_final.k_shift_fractional_bits
    {
        return Err(PeriodicConvergenceError::AxisBridgeMismatch);
    }
    let bridge_delta = (energy(first) - energy(cutoff_final)).abs();
    if bridge_delta > tolerance {
        return Err(PeriodicConvergenceError::AxisBridgeEnergyMismatch);
    }
    for pair in series.windows(2) {
        let mut strict = false;
        for axis in 0..3 {
            if pair[1].k_grid[axis] < pair[0].k_grid[axis] {
                return Err(PeriodicConvergenceError::KPointGridNotIncreasing);
            }
            strict |= pair[1].k_grid[axis] > pair[0].k_grid[axis];
        }
        if !strict {
            return Err(PeriodicConvergenceError::KPointGridNotIncreasing);
        }
    }
    require_final_two_deltas(series, tolerance)
}

fn require_final_two_deltas(
    series: &[PeriodicEnergyConvergenceSampleBinding],
    tolerance: f64,
) -> Result<(), PeriodicConvergenceError> {
    let n = series.len();
    let d1 = (energy(&series[n - 1]) - energy(&series[n - 2])).abs();
    let d2 = (energy(&series[n - 2]) - energy(&series[n - 3])).abs();
    if d1 > tolerance || d2 > tolerance {
        return Err(PeriodicConvergenceError::FinalEnergyPlateauNotReached);
    }
    Ok(())
}

fn energy(sample: &PeriodicEnergyConvergenceSampleBinding) -> f64 {
    f64::from_bits(sample.energy_per_atom_bits)
}

#[derive(Debug, Clone, PartialEq)]
pub enum PeriodicConvergenceError {
    Capability(CrystalCapabilityGateError),
    Leaf(CrystalExecutionError),
    ExternalEvidence(ExternalEvidenceError),
    EmptyProductionExecutionId,
    DuplicateStudyForExecution(String),
    StudySetMismatch,
    SameProductionContentDifferentIds,
    InconsistentProductionReuse(String),
    ProductionIdentityMismatch,
    EmptyCriterionId,
    InvalidEnergyTolerance,
    InsufficientAxisSamples,
    CriterionMissingFromProductionClaim,
    ProductionSolverMissing,
    DuplicateSampleId(String),
    SolverProfileDrift(String),
    CriterionMissingFromSampleClaim(String),
    DuplicateSampleExecutionId(String),
    DuplicateSampleExecutionContent,
    InvariantEvidenceMissingFromClaim {
        sample_id: String,
        evidence_id: String,
    },
    EnergyNotExecutionOutput(String),
    PseudopotentialNotExecutionDependency(String),
    InvariantEvidenceMissingDate(String),
    InvariantEvidencePostdatesExecution(String),
    ProductionIsNotFinalKPointSample,
    SampleOodNotPropagated,
    EmptySampleField,
    SampleIsNotPeriodicElectronicStructure,
    InvalidEnergy,
    InvalidCutoff,
    InvalidKGrid,
    InvalidKShift,
    NonResolutionSettingsDrift,
    CutoffAxisKPointDrift,
    CutoffAxisNotIncreasing,
    KPointAxisNonGridDrift,
    AxisBridgeMismatch,
    AxisBridgeEnergyMismatch,
    KPointGridNotIncreasing,
    FinalEnergyPlateauNotReached,
}

impl From<CrystalCapabilityGateError> for PeriodicConvergenceError {
    fn from(value: CrystalCapabilityGateError) -> Self {
        Self::Capability(value)
    }
}

impl From<CrystalExecutionError> for PeriodicConvergenceError {
    fn from(value: CrystalExecutionError) -> Self {
        Self::Leaf(value)
    }
}

impl From<ExternalEvidenceError> for PeriodicConvergenceError {
    fn from(value: ExternalEvidenceError) -> Self {
        Self::ExternalEvidence(value)
    }
}

impl fmt::Display for PeriodicConvergenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "periodic convergence gate rejected: {self:?}")
    }
}

impl std::error::Error for PeriodicConvergenceError {}
