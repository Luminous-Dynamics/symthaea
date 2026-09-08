// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Strict public boundary for periodic-energy convergence studies.
//!
//! The raw convergence gate validates each production energy independently. This
//! wrapper additionally prevents one intermediate numerical run or one energy
//! artifact from being reused across different phase studies and mistaken for
//! separate convergence evidence.

use std::collections::BTreeMap;
use std::fmt;

use symthaea_evidence_plane::external_receipt::ExternalEvidenceBundle;

use super::matter_crystal_evidence_binding::EvidenceBoundCrystalPhaseMatterClaim;
use super::matter_periodic_convergence_gate::bind_converged_crystal_execution_plan;

pub use super::matter_periodic_convergence_gate::{
    ConvergenceBoundCrystalExecutionPlan, PeriodicConvergenceError,
    PeriodicConvergenceInterpretation, PeriodicEnergyConvergenceSampleBinding,
    PeriodicEnergyConvergenceSampleReceipt, PeriodicEnergyConvergenceStudyBinding,
    PeriodicEnergyConvergenceStudyReceipt,
};
pub use super::matter_crystal_capability_gate::{
    CrystalCapabilityGateError, CrystalCapabilityInterpretation, CrystalClaimCapabilityBinding,
    CrystalClaimCapabilityReceipt, CrystalScientificCapability, MixedCrystalExecutionPlanReceipt,
};

pub fn bind_strictly_converged_crystal_execution_plan(
    evidence_bound: EvidenceBoundCrystalPhaseMatterClaim,
    plan_receipt: MixedCrystalExecutionPlanReceipt,
    capability_receipts: &[CrystalClaimCapabilityReceipt],
    studies: &[PeriodicEnergyConvergenceStudyReceipt],
    bundle: &ExternalEvidenceBundle,
) -> Result<ConvergenceBoundCrystalExecutionPlan, StrictPeriodicConvergenceError> {
    let bound = bind_converged_crystal_execution_plan(
        evidence_bound,
        plan_receipt,
        capability_receipts,
        studies,
        bundle,
    )?;
    validate_global_sample_uniqueness(&bound)?;
    Ok(bound)
}

fn validate_global_sample_uniqueness(
    bound: &ConvergenceBoundCrystalExecutionPlan,
) -> Result<(), StrictPeriodicConvergenceError> {
    let mut execution_ids = BTreeMap::<String, String>::new();
    let mut execution_digests = BTreeMap::<String, String>::new();
    let mut energy_digests = BTreeMap::<String, String>::new();

    for study in &bound.periodic_energy_studies {
        for sample in study.cutoff_series.iter().chain(study.kpoint_series.iter()) {
            let owner = format!("{}:{}", study.production_claim_id, sample.sample_id);

            if let Some(first) = execution_ids.insert(
                sample.execution.execution_evidence_id.clone(),
                owner.clone(),
            ) {
                return Err(StrictPeriodicConvergenceError::ExecutionReusedAcrossStudies {
                    first,
                    second: owner,
                });
            }

            let execution_digest = sample.execution.execution_content_sha256.as_str().to_string();
            if let Some(first) = execution_digests.insert(execution_digest, owner.clone()) {
                return Err(StrictPeriodicConvergenceError::ExecutionContentReusedAcrossStudies {
                    first,
                    second: owner,
                });
            }

            let energy_digest = sample.energy_output_sha256.as_str().to_string();
            if let Some(first) = energy_digests.insert(energy_digest, owner.clone()) {
                return Err(StrictPeriodicConvergenceError::EnergyOutputReusedAcrossStudies {
                    first,
                    second: owner,
                });
            }
        }
    }
    Ok(())
}

#[derive(Debug, Clone, PartialEq)]
pub enum StrictPeriodicConvergenceError {
    Convergence(PeriodicConvergenceError),
    ExecutionReusedAcrossStudies { first: String, second: String },
    ExecutionContentReusedAcrossStudies { first: String, second: String },
    EnergyOutputReusedAcrossStudies { first: String, second: String },
}

impl From<PeriodicConvergenceError> for StrictPeriodicConvergenceError {
    fn from(value: PeriodicConvergenceError) -> Self {
        Self::Convergence(value)
    }
}

impl fmt::Display for StrictPeriodicConvergenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "strict periodic convergence rejected: {self:?}")
    }
}

impl std::error::Error for StrictPeriodicConvergenceError {}
