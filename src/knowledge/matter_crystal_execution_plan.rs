// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Mixed execution plan for the four required crystal-science derivations.
//!
//! The crystal admission slots are not assumed to map one-to-one to processes:
//! periodic electronic structure is one canonical leaf, structural relaxation is
//! a restart-aware graph, phase competition is a many-energy graph, and lattice
//! dynamics is a batched q-point graph.
//!
//! A periodic electronic-structure execution may be reused as the target-phase
//! energy in the phase graph only when both the execution binding and Matter
//! claim are exactly identical. That reuse is recorded explicitly and is never
//! interpreted as independent confirmation. All other cross-slot solver-execution
//! reuse fails closed in v1.

use std::collections::BTreeMap;
use std::fmt;

use symthaea_epistemic_types::MatterClaim;
use symthaea_evidence_plane::external_receipt::{
    DeclaredChronologyInterpretation, EvidenceReferenceInterpretation, ExternalEvidenceBundle,
};

use super::matter_crystal_evidence_binding::{
    bind_crystal_phase_evidence, CrystalEvidenceBindingError, EvidenceBoundCrystalPhaseMatterClaim,
};
use super::matter_crystal_execution::{
    CrystalExecutionError, CrystalSolverExecutionBinding, CrystalSolverExecutionReceipt,
    CrystalSolverTask,
};
use super::matter_crystal_leaf_binding::bind_crystal_solver_leaf;
use super::matter_crystal_phase::CrystalPhaseEvidenceSlot;
use super::matter_lattice_dynamics_binding::{
    bind_lattice_dynamics_execution_graph, LatticeDynamicsExecutionGraphBinding,
    LatticeDynamicsExecutionGraphReceipt, LatticeDynamicsGraphError,
};
use super::matter_phase_competition_graph::{
    bind_phase_competition_execution_graph, PhaseCompetitionExecutionGraphBinding,
    PhaseCompetitionExecutionGraphReceipt, PhaseCompetitionGraphError, PhaseEnergyRole,
};
use super::matter_relaxation_execution_binding::{
    bind_structural_relaxation_execution_graph, StrictRelaxationExecutionError,
    StructuralRelaxationExecutionGraphBinding, StructuralRelaxationExecutionGraphReceipt,
};

#[derive(Debug, Clone, PartialEq)]
pub struct MixedCrystalExecutionPlanReceipt {
    pub periodic_execution: CrystalSolverExecutionReceipt,
    pub relaxation_graph: StructuralRelaxationExecutionGraphReceipt,
    pub phase_competition_graph: PhaseCompetitionExecutionGraphReceipt,
    pub lattice_dynamics_graph: LatticeDynamicsExecutionGraphReceipt,
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum CrystalExecutionUseRole {
    PeriodicElectronicStructure,
    PhaseTargetEnergy { phase_id: String },
    PhaseCompetitorEnergy { phase_id: String },
    RelaxationStep { step_index: u32 },
    LatticeDynamicsBatch { batch_id: String },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SharedCrystalExecutionReuse {
    pub execution_evidence_id: String,
    pub roles: Vec<CrystalExecutionUseRole>,
    /// Explicitly false: reuse is lineage sharing, not independent confirmation.
    pub independent_confirmation: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MixedCrystalExecutionPlanBinding {
    pub evidence_bound: EvidenceBoundCrystalPhaseMatterClaim,
    pub periodic_claim: MatterClaim,
    pub periodic_execution: CrystalSolverExecutionBinding,
    pub relaxation: StructuralRelaxationExecutionGraphBinding,
    pub phase_competition: PhaseCompetitionExecutionGraphBinding,
    pub lattice_dynamics: LatticeDynamicsExecutionGraphBinding,
    pub shared_execution_reuse: Vec<SharedCrystalExecutionReuse>,
    pub reference_interpretation: EvidenceReferenceInterpretation,
    pub chronology_interpretation: DeclaredChronologyInterpretation,
}

pub fn bind_mixed_crystal_execution_plan(
    evidence_bound: EvidenceBoundCrystalPhaseMatterClaim,
    plan: MixedCrystalExecutionPlanReceipt,
    bundle: &ExternalEvidenceBundle,
) -> Result<MixedCrystalExecutionPlanBinding, MixedCrystalExecutionPlanError> {
    let rebound = bind_crystal_phase_evidence(evidence_bound.admitted.clone(), bundle)?;
    if rebound != evidence_bound {
        return Err(MixedCrystalExecutionPlanError::ParentEvidenceBindingMismatch);
    }

    let periodic_claim = evidence_bound
        .admitted
        .crystal_evidence
        .iter()
        .find(|snapshot| snapshot.slot == CrystalPhaseEvidenceSlot::PeriodicElectronicStructure)
        .ok_or(MixedCrystalExecutionPlanError::MissingPeriodicClaim)?
        .claim
        .clone();
    if plan.periodic_execution.task != CrystalSolverTask::PeriodicElectronicStructure {
        return Err(MixedCrystalExecutionPlanError::PeriodicReceiptHasWrongTask);
    }
    let periodic_execution =
        bind_crystal_solver_leaf(&plan.periodic_execution, &periodic_claim, bundle)?;

    let relaxation = bind_structural_relaxation_execution_graph(
        evidence_bound.clone(),
        plan.relaxation_graph,
        bundle,
    )?;
    let phase_competition = bind_phase_competition_execution_graph(
        evidence_bound.clone(),
        plan.phase_competition_graph,
        bundle,
    )?;
    let lattice_dynamics = bind_lattice_dynamics_execution_graph(
        evidence_bound.clone(),
        plan.lattice_dynamics_graph,
        bundle,
    )?;

    for parent in [
        &relaxation.evidence_bound,
        &phase_competition.evidence_bound,
        &lattice_dynamics.evidence_bound,
    ] {
        if parent != &evidence_bound {
            return Err(MixedCrystalExecutionPlanError::ChildParentMismatch);
        }
    }

    let shared_execution_reuse = validate_cross_slot_execution_reuse(
        &periodic_claim,
        &periodic_execution,
        &relaxation,
        &phase_competition,
        &lattice_dynamics,
    )?;

    Ok(MixedCrystalExecutionPlanBinding {
        evidence_bound,
        periodic_claim,
        periodic_execution,
        relaxation,
        phase_competition,
        lattice_dynamics,
        shared_execution_reuse,
        reference_interpretation: EvidenceReferenceInterpretation::ReferenceOnly,
        chronology_interpretation: DeclaredChronologyInterpretation::DeclaredChronologyOnly,
    })
}

#[derive(Debug, Clone)]
struct ExecutionUse<'a> {
    role: CrystalExecutionUseRole,
    claim: &'a MatterClaim,
    execution: &'a CrystalSolverExecutionBinding,
}

fn validate_cross_slot_execution_reuse(
    periodic_claim: &MatterClaim,
    periodic: &CrystalSolverExecutionBinding,
    relaxation: &StructuralRelaxationExecutionGraphBinding,
    phase: &PhaseCompetitionExecutionGraphBinding,
    lattice: &LatticeDynamicsExecutionGraphBinding,
) -> Result<Vec<SharedCrystalExecutionReuse>, MixedCrystalExecutionPlanError> {
    let mut uses = vec![ExecutionUse {
        role: CrystalExecutionUseRole::PeriodicElectronicStructure,
        claim: periodic_claim,
        execution: periodic,
    }];

    for leaf in &phase.leaves {
        uses.push(ExecutionUse {
            role: match leaf.role {
                PhaseEnergyRole::TargetPhase => CrystalExecutionUseRole::PhaseTargetEnergy {
                    phase_id: leaf.phase_id.clone(),
                },
                PhaseEnergyRole::CompetingPhase => {
                    CrystalExecutionUseRole::PhaseCompetitorEnergy {
                        phase_id: leaf.phase_id.clone(),
                    }
                }
            },
            claim: &leaf.claim,
            execution: &leaf.execution,
        });
    }
    for step in &relaxation.steps {
        uses.push(ExecutionUse {
            role: CrystalExecutionUseRole::RelaxationStep {
                step_index: step.step_index,
            },
            claim: &step.claim,
            execution: &step.execution,
        });
    }
    for batch in &lattice.batches {
        uses.push(ExecutionUse {
            role: CrystalExecutionUseRole::LatticeDynamicsBatch {
                batch_id: batch.batch_id.clone(),
            },
            claim: &batch.claim,
            execution: &batch.execution,
        });
    }

    let mut by_execution_id: BTreeMap<String, Vec<&ExecutionUse<'_>>> = BTreeMap::new();
    let mut by_execution_digest: BTreeMap<String, Vec<&ExecutionUse<'_>>> = BTreeMap::new();
    for usage in &uses {
        by_execution_id
            .entry(usage.execution.execution_evidence_id.clone())
            .or_default()
            .push(usage);
        by_execution_digest
            .entry(usage.execution.execution_content_sha256.as_str().to_string())
            .or_default()
            .push(usage);
    }

    // Distinct evidence ids cannot hide reuse of the same execution content.
    for digest_uses in by_execution_digest.values() {
        if digest_uses.len() > 1 {
            let first_id = &digest_uses[0].execution.execution_evidence_id;
            if digest_uses
                .iter()
                .any(|usage| &usage.execution.execution_evidence_id != first_id)
            {
                return Err(MixedCrystalExecutionPlanError::SameExecutionContentDifferentIds);
            }
        }
    }

    let mut shared = Vec::new();
    for (execution_id, id_uses) in by_execution_id {
        if id_uses.len() == 1 {
            continue;
        }
        if !allowed_periodic_phase_target_reuse(&id_uses) {
            return Err(MixedCrystalExecutionPlanError::ForbiddenCrossSlotExecutionReuse {
                execution_id,
                roles: id_uses.iter().map(|usage| usage.role.clone()).collect(),
            });
        }
        shared.push(SharedCrystalExecutionReuse {
            execution_evidence_id: execution_id,
            roles: id_uses.iter().map(|usage| usage.role.clone()).collect(),
            independent_confirmation: false,
        });
    }
    Ok(shared)
}

fn allowed_periodic_phase_target_reuse(uses: &[&ExecutionUse<'_>]) -> bool {
    if uses.len() != 2 {
        return false;
    }
    let periodic = uses
        .iter()
        .find(|usage| matches!(usage.role, CrystalExecutionUseRole::PeriodicElectronicStructure));
    let target = uses.iter().find(|usage| {
        matches!(usage.role, CrystalExecutionUseRole::PhaseTargetEnergy { .. })
    });
    let (Some(periodic), Some(target)) = (periodic, target) else {
        return false;
    };

    periodic.claim == target.claim && periodic.execution == target.execution
}

#[derive(Debug, Clone, PartialEq)]
pub enum MixedCrystalExecutionPlanError {
    CrystalEvidence(CrystalEvidenceBindingError),
    Leaf(CrystalExecutionError),
    Relaxation(StrictRelaxationExecutionError),
    PhaseCompetition(PhaseCompetitionGraphError),
    LatticeDynamics(LatticeDynamicsGraphError),
    ParentEvidenceBindingMismatch,
    MissingPeriodicClaim,
    PeriodicReceiptHasWrongTask,
    ChildParentMismatch,
    SameExecutionContentDifferentIds,
    ForbiddenCrossSlotExecutionReuse {
        execution_id: String,
        roles: Vec<CrystalExecutionUseRole>,
    },
}

impl From<CrystalEvidenceBindingError> for MixedCrystalExecutionPlanError {
    fn from(value: CrystalEvidenceBindingError) -> Self {
        Self::CrystalEvidence(value)
    }
}
impl From<CrystalExecutionError> for MixedCrystalExecutionPlanError {
    fn from(value: CrystalExecutionError) -> Self {
        Self::Leaf(value)
    }
}
impl From<StrictRelaxationExecutionError> for MixedCrystalExecutionPlanError {
    fn from(value: StrictRelaxationExecutionError) -> Self {
        Self::Relaxation(value)
    }
}
impl From<PhaseCompetitionGraphError> for MixedCrystalExecutionPlanError {
    fn from(value: PhaseCompetitionGraphError) -> Self {
        Self::PhaseCompetition(value)
    }
}
impl From<LatticeDynamicsGraphError> for MixedCrystalExecutionPlanError {
    fn from(value: LatticeDynamicsGraphError) -> Self {
        Self::LatticeDynamics(value)
    }
}

impl fmt::Display for MixedCrystalExecutionPlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "mixed crystal execution plan rejected: {self:?}")
    }
}

impl std::error::Error for MixedCrystalExecutionPlanError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_periodic_target_reuse_is_not_independent_confirmation() {
        let role_a = CrystalExecutionUseRole::PeriodicElectronicStructure;
        let role_b = CrystalExecutionUseRole::PhaseTargetEnergy {
            phase_id: "target".into(),
        };
        let reuse = SharedCrystalExecutionReuse {
            execution_evidence_id: "exec".into(),
            roles: vec![role_a, role_b],
            independent_confirmation: false,
        };
        assert!(!reuse.independent_confirmation);
    }

    #[test]
    fn reuse_roles_are_explicit_not_confidence_weighted() {
        assert_ne!(
            CrystalExecutionUseRole::RelaxationStep { step_index: 0 },
            CrystalExecutionUseRole::LatticeDynamicsBatch {
                batch_id: "batch".into()
            }
        );
    }
}
