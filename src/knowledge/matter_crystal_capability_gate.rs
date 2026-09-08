// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Strict public capability boundary for mixed crystal execution plans.
//!
//! A solver name, successful process, or plausible numerical result does not
//! establish that the implementation supports the exact scientific task being
//! claimed. This layer requires each participating Matter claim to retain a
//! canonical capability-manifest artifact tied to its exact solver identity.
//!
//! A capability manifest remains a declaration. This boundary does not inspect
//! or authenticate the manifest bytes and therefore never upgrades E/N/M.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use symthaea_epistemic_types::MatterClaim;
use symthaea_evidence_plane::external_receipt::{
    ClaimedUtcDate, EvidenceReferenceInterpretation, EvidenceRole, ExternalEvidenceBundle,
    ExternalEvidenceError, Sha256Digest,
};

use super::matter_crystal_evidence_binding::EvidenceBoundCrystalPhaseMatterClaim;
use super::matter_crystal_execution_plan::bind_mixed_crystal_execution_plan;
use super::matter_crystal_phase::CrystalPhaseEvidenceSlot;

pub use super::matter_crystal_execution_plan::{
    CrystalExecutionUseRole, MixedCrystalExecutionPlanBinding, MixedCrystalExecutionPlanError,
    MixedCrystalExecutionPlanReceipt, SharedCrystalExecutionReuse,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum CrystalScientificCapability {
    PeriodicElectronicStructure,
    StructuralRelaxation,
    ThermodynamicPhaseCompetition,
    LatticeDynamics,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CrystalCapabilityInterpretation {
    DeclaredCapabilityOnly,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CrystalClaimCapabilityReceipt {
    pub claim_id: String,
    pub capability: CrystalScientificCapability,
    pub solver_name: String,
    pub solver_version: String,
    pub solver_method: String,
    pub capability_manifest_evidence_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CrystalClaimCapabilityBinding {
    pub claim_id: String,
    pub capability: CrystalScientificCapability,
    pub capability_manifest_evidence_id: String,
    pub capability_manifest_sha256: Sha256Digest,
    pub capability_manifest_claimed_date: ClaimedUtcDate,
    pub required_not_after_date: ClaimedUtcDate,
    pub reference_interpretation: EvidenceReferenceInterpretation,
    pub capability_interpretation: CrystalCapabilityInterpretation,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CapabilityBoundCrystalExecutionPlan {
    pub plan: MixedCrystalExecutionPlanBinding,
    pub capability_bindings: Vec<CrystalClaimCapabilityBinding>,
    pub reference_interpretation: EvidenceReferenceInterpretation,
    pub capability_interpretation: CrystalCapabilityInterpretation,
}

#[derive(Debug, Clone)]
struct Requirement {
    claim: MatterClaim,
    capability: CrystalScientificCapability,
    not_after: ClaimedUtcDate,
}

pub fn bind_capability_gated_crystal_execution_plan(
    evidence_bound: EvidenceBoundCrystalPhaseMatterClaim,
    plan_receipt: MixedCrystalExecutionPlanReceipt,
    receipts: &[CrystalClaimCapabilityReceipt],
    bundle: &ExternalEvidenceBundle,
) -> Result<CapabilityBoundCrystalExecutionPlan, CrystalCapabilityGateError> {
    let plan = bind_mixed_crystal_execution_plan(evidence_bound, plan_receipt, bundle)?;
    let requirements = collect_requirements(&plan)?;

    let mut supplied = BTreeMap::new();
    for receipt in receipts {
        validate_receipt(receipt)?;
        let key = (receipt.claim_id.clone(), receipt.capability);
        if supplied.insert(key.clone(), receipt).is_some() {
            return Err(CrystalCapabilityGateError::DuplicateReceipt(key));
        }
    }

    let expected: BTreeSet<_> = requirements.keys().cloned().collect();
    let actual: BTreeSet<_> = supplied.keys().cloned().collect();
    if expected != actual {
        return Err(CrystalCapabilityGateError::ReceiptSetMismatch);
    }

    let mut bindings = Vec::with_capacity(requirements.len());
    for (key, requirement) in requirements {
        let receipt = supplied
            .get(&key)
            .expect("capability key sets were checked for equality");
        bindings.push(bind_one(requirement, receipt, bundle)?);
    }
    bindings.sort_by(|a, b| {
        a.claim_id
            .cmp(&b.claim_id)
            .then(a.capability.cmp(&b.capability))
    });

    Ok(CapabilityBoundCrystalExecutionPlan {
        plan,
        capability_bindings: bindings,
        reference_interpretation: EvidenceReferenceInterpretation::ReferenceOnly,
        capability_interpretation: CrystalCapabilityInterpretation::DeclaredCapabilityOnly,
    })
}

fn collect_requirements(
    plan: &MixedCrystalExecutionPlanBinding,
) -> Result<
    BTreeMap<(String, CrystalScientificCapability), Requirement>,
    CrystalCapabilityGateError,
> {
    let mut out = BTreeMap::new();

    insert_requirement(
        &mut out,
        &plan.periodic_claim,
        CrystalScientificCapability::PeriodicElectronicStructure,
        plan.periodic_execution.execution_claimed_date,
    )?;

    for step in &plan.relaxation.steps {
        insert_requirement(
            &mut out,
            &step.claim,
            CrystalScientificCapability::StructuralRelaxation,
            step.execution.execution_claimed_date,
        )?;
    }

    for leaf in &plan.phase_competition.leaves {
        insert_requirement(
            &mut out,
            &leaf.claim,
            CrystalScientificCapability::PeriodicElectronicStructure,
            leaf.execution.execution_claimed_date,
        )?;
    }
    insert_requirement(
        &mut out,
        claim_for_slot(
            plan,
            CrystalPhaseEvidenceSlot::ThermodynamicPhaseCompetition,
        )?,
        CrystalScientificCapability::ThermodynamicPhaseCompetition,
        plan.phase_competition.aggregation.output_claimed_date,
    )?;

    for batch in &plan.lattice_dynamics.batches {
        insert_requirement(
            &mut out,
            &batch.claim,
            CrystalScientificCapability::LatticeDynamics,
            batch.execution.execution_claimed_date,
        )?;
    }
    insert_requirement(
        &mut out,
        claim_for_slot(plan, CrystalPhaseEvidenceSlot::DynamicalStability)?,
        CrystalScientificCapability::LatticeDynamics,
        plan.lattice_dynamics.aggregation.output_claimed_date,
    )?;

    Ok(out)
}

fn claim_for_slot(
    plan: &MixedCrystalExecutionPlanBinding,
    slot: CrystalPhaseEvidenceSlot,
) -> Result<&MatterClaim, CrystalCapabilityGateError> {
    plan.evidence_bound
        .admitted
        .crystal_evidence
        .iter()
        .find(|snapshot| snapshot.slot == slot)
        .map(|snapshot| &snapshot.claim)
        .ok_or(CrystalCapabilityGateError::MissingClaimForSlot(slot))
}

fn insert_requirement(
    out: &mut BTreeMap<(String, CrystalScientificCapability), Requirement>,
    claim: &MatterClaim,
    capability: CrystalScientificCapability,
    not_after: ClaimedUtcDate,
) -> Result<(), CrystalCapabilityGateError> {
    validate_solver(claim)?;
    let key = (claim.claim_id.clone(), capability);
    match out.get_mut(&key) {
        Some(existing) => {
            if existing.claim != *claim {
                return Err(CrystalCapabilityGateError::InconsistentReusedClaim(key));
            }
            if not_after < existing.not_after {
                existing.not_after = not_after;
            }
        }
        None => {
            out.insert(
                key,
                Requirement {
                    claim: claim.clone(),
                    capability,
                    not_after,
                },
            );
        }
    }
    Ok(())
}

fn validate_receipt(
    receipt: &CrystalClaimCapabilityReceipt,
) -> Result<(), CrystalCapabilityGateError> {
    for value in [
        receipt.claim_id.as_str(),
        receipt.solver_name.as_str(),
        receipt.solver_version.as_str(),
        receipt.solver_method.as_str(),
        receipt.capability_manifest_evidence_id.as_str(),
    ] {
        if value.trim().is_empty() {
            return Err(CrystalCapabilityGateError::EmptyReceiptField);
        }
    }
    Ok(())
}

fn validate_solver(claim: &MatterClaim) -> Result<(), CrystalCapabilityGateError> {
    let solver = claim
        .solver
        .as_ref()
        .ok_or_else(|| CrystalCapabilityGateError::MissingSolver(claim.claim_id.clone()))?;
    if solver.name.trim().is_empty()
        || solver.version.trim().is_empty()
        || solver.method.trim().is_empty()
    {
        return Err(CrystalCapabilityGateError::MalformedSolver(
            claim.claim_id.clone(),
        ));
    }
    Ok(())
}

fn bind_one(
    requirement: Requirement,
    receipt: &CrystalClaimCapabilityReceipt,
    bundle: &ExternalEvidenceBundle,
) -> Result<CrystalClaimCapabilityBinding, CrystalCapabilityGateError> {
    let solver = requirement
        .claim
        .solver
        .as_ref()
        .ok_or_else(|| CrystalCapabilityGateError::MissingSolver(requirement.claim.claim_id.clone()))?;

    if receipt.claim_id != requirement.claim.claim_id
        || receipt.capability != requirement.capability
        || receipt.solver_name != solver.name
        || receipt.solver_version != solver.version
        || receipt.solver_method != solver.method
    {
        return Err(CrystalCapabilityGateError::SolverCapabilityMismatch(
            requirement.claim.claim_id.clone(),
        ));
    }

    if !requirement
        .claim
        .evidence
        .iter()
        .any(|evidence| evidence.evidence_id == receipt.capability_manifest_evidence_id)
    {
        return Err(CrystalCapabilityGateError::ManifestNotRetainedByClaim(
            requirement.claim.claim_id.clone(),
        ));
    }

    let manifest = bundle.require_role(
        &receipt.capability_manifest_evidence_id,
        EvidenceRole::ArtifactContent,
    )?;
    let date = manifest.claimed_utc_date.ok_or_else(|| {
        CrystalCapabilityGateError::ManifestMissingDate(
            receipt.capability_manifest_evidence_id.clone(),
        )
    })?;
    if date > requirement.not_after {
        return Err(CrystalCapabilityGateError::ManifestPostdatesUse {
            claim_id: requirement.claim.claim_id.clone(),
            capability: requirement.capability,
        });
    }

    Ok(CrystalClaimCapabilityBinding {
        claim_id: requirement.claim.claim_id,
        capability: requirement.capability,
        capability_manifest_evidence_id: manifest.id.as_str().to_string(),
        capability_manifest_sha256: manifest.content_sha256.clone(),
        capability_manifest_claimed_date: date,
        required_not_after_date: requirement.not_after,
        reference_interpretation: EvidenceReferenceInterpretation::ReferenceOnly,
        capability_interpretation: CrystalCapabilityInterpretation::DeclaredCapabilityOnly,
    })
}

#[derive(Debug, Clone, PartialEq)]
pub enum CrystalCapabilityGateError {
    MixedPlan(MixedCrystalExecutionPlanError),
    ExternalEvidence(ExternalEvidenceError),
    DuplicateReceipt((String, CrystalScientificCapability)),
    ReceiptSetMismatch,
    MissingClaimForSlot(CrystalPhaseEvidenceSlot),
    MissingSolver(String),
    MalformedSolver(String),
    InconsistentReusedClaim((String, CrystalScientificCapability)),
    EmptyReceiptField,
    SolverCapabilityMismatch(String),
    ManifestNotRetainedByClaim(String),
    ManifestMissingDate(String),
    ManifestPostdatesUse {
        claim_id: String,
        capability: CrystalScientificCapability,
    },
}

impl From<MixedCrystalExecutionPlanError> for CrystalCapabilityGateError {
    fn from(value: MixedCrystalExecutionPlanError) -> Self {
        Self::MixedPlan(value)
    }
}

impl From<ExternalEvidenceError> for CrystalCapabilityGateError {
    fn from(value: ExternalEvidenceError) -> Self {
        Self::ExternalEvidence(value)
    }
}

impl fmt::Display for CrystalCapabilityGateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "crystal capability gate rejected: {self:?}")
    }
}

impl std::error::Error for CrystalCapabilityGateError {}
