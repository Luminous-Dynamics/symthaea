// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Permit-bound end-to-end Reach preparation and finalization.
//!
//! This module is the first path that carries one fresh semantic/spatial
//! validation lineage into the existing whole-body dynamics hierarchy and then
//! through typed goal authority plus the final safety projector.
//!
//! The prepared object intentionally owns two live borrows at once:
//! - an opaque spatial permit, which keeps the guarded validation cycle alive;
//! - the exact mutable `HumanoidExecutionPipeline` instance used to prepare the
//!   command, which preserves safety-projector history until finalization.
//!
//! As a result, callers cannot revalidate the skill or substitute/reset the
//! execution pipeline while a prepared Reach command exists. The object is
//! one-shot: finalization consumes it.

use crate::cartesian_hand_reference::{
    HumanoidCartesianHandReferenceFailure, HumanoidCartesianHandReferenceProfile,
    HumanoidCartesianHandReferenceReport, lower_permitted_cartesian_hand_reference,
};
use crate::contact::ContactFrame;
use crate::dynamics::RigidBodyDynamicsProvider;
use crate::execution::{HumanoidExecutionPipeline, HumanoidExecutionResult, HumanoidPreparedCommand};
use crate::execution_authority_scope::{
    HumanoidExecutionAuthorityScopeValidationFailure, HumanoidExecutionPurpose,
    HumanoidQualificationAuthorityBasis, HumanoidScopedSkillAuthorityReceipt,
};
use crate::floating_base::FloatingBaseDynamicsProvider;
use crate::frozen_dynamics::FrozenHumanoidDynamicsEnvironment;
use crate::full_dynamics::FullRigidBodyDynamicsProvider;
use crate::morphology::HandSide;
use crate::skill_runtime::HumanoidSkillIntent;
use crate::spatial_goal::HumanoidSpatiallyBoundSkillPermit;
use crate::spatial_goal_identity::humanoid_spatial_goal_fingerprint;
use crate::terrain::TerrainProbe;
use crate::types::{
    ActuationMode, HumanoidCommand, HumanoidPdGains, HumanoidState, HumanoidTask,
    pd_standing_baseline,
};
use crate::whole_body_intent::{
    HumanoidWholeBodyInvariantIR, HumanoidWholeBodyMotionIntent, HumanoidWholeBodyObjectiveIR,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidPermittedReachPreparationFailure {
    PipelineMorphologyMismatch,
    PermitIntentActuationMismatch,
    InvalidControlPeriod,
    InvalidPreparationTime,
    InvalidPdGains,
    InvalidReachIntent,
    InvalidSpatialGoalIdentity,
    MissingFullDynamics,
    CartesianReference(HumanoidCartesianHandReferenceFailure),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidPermittedReachFinalizationFailure {
    InvalidFinalizationTime,
    FinalizationBeforePreparation,
    AuthorityScope(HumanoidExecutionAuthorityScopeValidationFailure),
}

#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidFrozenDynamicsLineage {
    pub rigid_model_id: Option<String>,
    pub rigid_sampled_at_s: Option<f64>,
    pub full_model_id: String,
    pub full_sampled_at_s: f64,
    pub floating_model_id: Option<String>,
    pub floating_sampled_at_s: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidPermittedReachPreparationReport {
    pub validation_epoch: u64,
    pub prepared_at_s: f64,
    pub goal_id: String,
    pub spatial_goal_fingerprint: u64,
    pub hand: HandSide,
    pub target_world_m: [f64; 3],
    pub target_root_m: [f64; 3],
    pub workspace_utilization_sq: f64,
    pub dynamics: HumanoidFrozenDynamicsLineage,
    pub cartesian_reference: HumanoidCartesianHandReferenceReport,
}

#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidReachAuthorityReceiptAudit {
    pub receipt_fingerprint: u64,
    pub scope_fingerprint: u64,
    pub scope_id: String,
    pub execution_purpose: HumanoidExecutionPurpose,
    pub qualification_basis: HumanoidQualificationAuthorityBasis,
    pub validation_epoch: u64,
    pub issued_at_s: f64,
    pub valid_until_s: f64,
    pub finalized_at_s: f64,
    pub requirement_subject_fingerprints: Vec<u64>,
    pub operator_evidence_id: String,
    pub qualification_evidence_id: String,
    pub physical_evidence_id: String,
    pub epistemic_evidence_id: String,
    pub cognitive_evidence_id: String,
    pub operator_scale: f32,
    pub qualification_scale: f32,
    pub physical_scale: f32,
    pub epistemic_scale: f32,
    pub cognitive_scale: f32,
}

#[derive(Debug, Clone)]
pub struct HumanoidPermittedReachExecutionResult {
    pub execution: HumanoidExecutionResult,
    pub preparation: HumanoidPermittedReachPreparationReport,
    pub authority_receipt: HumanoidReachAuthorityReceiptAudit,
}

pub struct HumanoidPermittedReachPreparedCommand<'pipeline, 'permit> {
    pipeline: &'pipeline mut HumanoidExecutionPipeline,
    permit: HumanoidSpatiallyBoundSkillPermit<'permit>,
    prepared: HumanoidPreparedCommand,
    state: HumanoidState,
    actuation_mode: ActuationMode,
    dt: f64,
    report: HumanoidPermittedReachPreparationReport,
}

impl<'pipeline, 'permit> HumanoidPermittedReachPreparedCommand<'pipeline, 'permit> {
    pub fn report(&self) -> &HumanoidPermittedReachPreparationReport {
        &self.report
    }

    pub fn hierarchy_report(&self) -> &crate::hierarchical::HierarchicalControlReport {
        self.prepared.hierarchy_report()
    }

    /// Consume the prepared command only after purpose-scoped authority proves it
    /// is bound to the same live permit and the correct qualification/operational
    /// context. A trial-protocol receipt cannot masquerade as operational authority.
    pub fn finalize(
        self,
        authority_receipt: HumanoidScopedSkillAuthorityReceipt,
        now_s: f64,
    ) -> Result<HumanoidPermittedReachExecutionResult, HumanoidPermittedReachFinalizationFailure> {
        if !now_s.is_finite() || now_s < 0.0 {
            return Err(HumanoidPermittedReachFinalizationFailure::InvalidFinalizationTime);
        }
        if now_s < self.report.prepared_at_s {
            return Err(HumanoidPermittedReachFinalizationFailure::FinalizationBeforePreparation);
        }
        authority_receipt
            .validate_for_permit(self.permit.semantic(), now_s)
            .map_err(HumanoidPermittedReachFinalizationFailure::AuthorityScope)?;

        let source = authority_receipt.source_evidence();
        let authority_audit = HumanoidReachAuthorityReceiptAudit {
            receipt_fingerprint: authority_receipt.receipt_fingerprint(),
            scope_fingerprint: authority_receipt.scope_fingerprint(),
            scope_id: authority_receipt.scope_id().to_string(),
            execution_purpose: authority_receipt.purpose(),
            qualification_basis: authority_receipt.qualification_basis(),
            validation_epoch: authority_receipt.validation_epoch(),
            issued_at_s: authority_receipt.issued_at_s(),
            valid_until_s: authority_receipt.valid_until_s(),
            finalized_at_s: now_s,
            requirement_subject_fingerprints: authority_receipt
                .requirement_subject_fingerprints()
                .to_vec(),
            operator_evidence_id: source.operator.evidence_id.clone(),
            qualification_evidence_id: source.qualification.evidence_id.clone(),
            physical_evidence_id: source.physical.evidence_id.clone(),
            epistemic_evidence_id: source.epistemic.evidence_id.clone(),
            cognitive_evidence_id: source.cognitive.evidence_id.clone(),
            operator_scale: source.operator.scale,
            qualification_scale: source.qualification.scale,
            physical_scale: source.physical.scale,
            epistemic_scale: source.epistemic.scale,
            cognitive_scale: source.cognitive.scale,
        };
        let authority = authority_receipt.authority_envelope();

        let Self {
            pipeline,
            permit: _,
            prepared,
            state,
            actuation_mode,
            dt,
            report,
        } = self;
        let execution = pipeline.finalize_prepared(prepared, &state, authority, actuation_mode, dt);
        Ok(HumanoidPermittedReachExecutionResult {
            execution,
            preparation: report,
            authority_receipt: authority_audit,
        })
    }
}

#[allow(clippy::too_many_arguments)]
pub fn prepare_permitted_reach<'pipeline, 'permit, T>(
    pipeline: &'pipeline mut HumanoidExecutionPipeline,
    permit: HumanoidSpatiallyBoundSkillPermit<'permit>,
    intent: &HumanoidWholeBodyMotionIntent,
    state: &HumanoidState,
    contacts: &ContactFrame,
    environment: &T,
    cartesian_profile: &HumanoidCartesianHandReferenceProfile,
    pd_gains: &HumanoidPdGains,
    now_s: f64,
    dt: f64,
) -> Result<HumanoidPermittedReachPreparedCommand<'pipeline, 'permit>, HumanoidPermittedReachPreparationFailure>
where
    T: TerrainProbe
        + RigidBodyDynamicsProvider
        + FullRigidBodyDynamicsProvider
        + FloatingBaseDynamicsProvider
        + ?Sized,
{
    if pipeline.morphology() != intent.morphology
        || pipeline.morphology() != permit.semantic().morphology()
    {
        return Err(HumanoidPermittedReachPreparationFailure::PipelineMorphologyMismatch);
    }
    if intent.actuation_mode != permit.semantic().actuation_mode() {
        return Err(HumanoidPermittedReachPreparationFailure::PermitIntentActuationMismatch);
    }
    if !dt.is_finite() || dt <= 0.0 {
        return Err(HumanoidPermittedReachPreparationFailure::InvalidControlPeriod);
    }
    if !now_s.is_finite() || now_s < 0.0 {
        return Err(HumanoidPermittedReachPreparationFailure::InvalidPreparationTime);
    }
    if !valid_pd_gains(pd_gains, pipeline.morphology().num_actuators()) {
        return Err(HumanoidPermittedReachPreparationFailure::InvalidPdGains);
    }
    if !valid_reach_intent(intent) {
        return Err(HumanoidPermittedReachPreparationFailure::InvalidReachIntent);
    }

    let spatial_goal_fingerprint = humanoid_spatial_goal_fingerprint(permit.goal());
    if spatial_goal_fingerprint == 0 {
        return Err(HumanoidPermittedReachPreparationFailure::InvalidSpatialGoalIdentity);
    }
    if !permit.workspace_utilization_sq().is_finite() || permit.workspace_utilization_sq() < 0.0 {
        return Err(HumanoidPermittedReachPreparationFailure::InvalidSpatialGoalIdentity);
    }

    let frozen = FrozenHumanoidDynamicsEnvironment::capture(environment, state, contacts);
    let dynamics = frozen
        .full_snapshot()
        .ok_or(HumanoidPermittedReachPreparationFailure::MissingFullDynamics)?;
    let cartesian = lower_permitted_cartesian_hand_reference(
        &permit,
        intent,
        state,
        dynamics,
        cartesian_profile,
        now_s,
    )
    .map_err(HumanoidPermittedReachPreparationFailure::CartesianReference)?;

    let n = pipeline.morphology().num_actuators();
    let mut baseline = pd_standing_baseline(state, pd_gains);
    if baseline.num_actuators() != n || cartesian.correction.num_actuators() != n {
        return Err(HumanoidPermittedReachPreparationFailure::InvalidPdGains);
    }
    for (value, correction) in baseline
        .torques
        .iter_mut()
        .zip(cartesian.correction.torques.iter())
    {
        *value = (*value + *correction).clamp(-1.0, 1.0);
    }

    let learned_residual = HumanoidCommand::zero_for(n);
    let prepared = pipeline.prepare_with_environment(
        HumanoidTask::Reach,
        state,
        contacts,
        &frozen,
        &baseline,
        &learned_residual,
        1.0,
        0.0,
    );

    let dynamics_lineage = HumanoidFrozenDynamicsLineage {
        rigid_model_id: frozen.rigid_snapshot().map(|snapshot| snapshot.model_id.clone()),
        rigid_sampled_at_s: frozen.rigid_snapshot().map(|snapshot| snapshot.sampled_at_s),
        full_model_id: dynamics.model_id.clone(),
        full_sampled_at_s: dynamics.sampled_at_s,
        floating_model_id: frozen
            .floating_snapshot()
            .map(|snapshot| snapshot.model_id.clone()),
        floating_sampled_at_s: frozen
            .floating_snapshot()
            .map(|snapshot| snapshot.sampled_at_s),
    };
    let report = HumanoidPermittedReachPreparationReport {
        validation_epoch: permit.semantic().epoch(),
        prepared_at_s: now_s,
        goal_id: permit.goal().goal_id.clone(),
        spatial_goal_fingerprint,
        hand: permit.goal().hand,
        target_world_m: permit.goal().target_world_m,
        target_root_m: permit.target_root_m(),
        workspace_utilization_sq: permit.workspace_utilization_sq(),
        dynamics: dynamics_lineage,
        cartesian_reference: cartesian.report,
    };

    Ok(HumanoidPermittedReachPreparedCommand {
        pipeline,
        permit,
        prepared,
        state: state.clone(),
        actuation_mode: intent.actuation_mode,
        dt,
        report,
    })
}

fn valid_pd_gains(gains: &HumanoidPdGains, actuators: usize) -> bool {
    gains.kp.len() == actuators
        && gains.kd.len() == actuators
        && gains.kp.iter().all(|value| value.is_finite() && *value >= 0.0)
        && gains.kd.iter().all(|value| value.is_finite() && *value >= 0.0)
}

fn valid_reach_intent(intent: &HumanoidWholeBodyMotionIntent) -> bool {
    if !matches!(intent.source_skill, HumanoidSkillIntent::Reach { .. })
        || intent.validation_epoch == 0
        || intent.spatial_goal_id.as_deref().is_none_or(str::is_empty)
    {
        return false;
    }

    let mut upright = 0usize;
    let mut end_effector = 0usize;
    for objective in &intent.objectives {
        match objective {
            HumanoidWholeBodyObjectiveIR::UprightPosture => upright += 1,
            HumanoidWholeBodyObjectiveIR::EndEffectorTarget {
                target_root_m,
                maximum_speed_mps,
                ..
            } => {
                end_effector += 1;
                if !target_root_m.iter().all(|value| value.is_finite())
                    || !maximum_speed_mps.is_finite()
                    || *maximum_speed_mps < 0.0
                {
                    return false;
                }
            }
            HumanoidWholeBodyObjectiveIR::LocomotionVelocity { .. }
            | HumanoidWholeBodyObjectiveIR::ObjectContact { .. }
            | HumanoidWholeBodyObjectiveIR::HumanContact { .. }
            | HumanoidWholeBodyObjectiveIR::Payload { .. } => return false,
        }
    }
    if upright != 1 || end_effector != 1 {
        return false;
    }

    let required = [
        HumanoidWholeBodyInvariantIR::CapabilityEnvelopeRemainsAdmitted,
        HumanoidWholeBodyInvariantIR::QualificationSubjectRemainsStable,
        HumanoidWholeBodyInvariantIR::ProtectiveBehaviorMayPreemptGoal,
        HumanoidWholeBodyInvariantIR::SpatialGoalRemainsBound,
    ];
    if intent.invariants.len() != required.len() {
        return false;
    }
    required.into_iter().all(|needle| {
        intent
            .invariants
            .iter()
            .filter(|value| **value == needle)
            .count()
            == 1
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::morphology::HumanoidMorphology;
    use crate::types::ActuationMode;

    fn reach_intent() -> HumanoidWholeBodyMotionIntent {
        HumanoidWholeBodyMotionIntent {
            validation_epoch: 4,
            morphology: HumanoidMorphology::Dexterous53,
            actuation_mode: ActuationMode::NormalizedTorque,
            backend_profile_id: "reach-exec-test-v1".into(),
            source_skill: HumanoidSkillIntent::Reach {
                end_effector_speed_mps: 0.2,
            },
            requirement_subject_fingerprints: vec![42],
            spatial_goal_id: Some("cup-7".into()),
            objectives: vec![
                HumanoidWholeBodyObjectiveIR::UprightPosture,
                HumanoidWholeBodyObjectiveIR::EndEffectorTarget {
                    hand: crate::morphology::HandSide::Right,
                    target_root_m: [0.3, -0.2, 0.2],
                    maximum_speed_mps: 0.2,
                },
            ],
            invariants: vec![
                HumanoidWholeBodyInvariantIR::CapabilityEnvelopeRemainsAdmitted,
                HumanoidWholeBodyInvariantIR::QualificationSubjectRemainsStable,
                HumanoidWholeBodyInvariantIR::ProtectiveBehaviorMayPreemptGoal,
                HumanoidWholeBodyInvariantIR::SpatialGoalRemainsBound,
            ],
        }
    }

    #[test]
    fn exact_reach_shape_is_accepted() {
        assert!(valid_reach_intent(&reach_intent()));
    }

    #[test]
    fn contact_objective_cannot_be_erased_by_reach_executor() {
        let mut intent = reach_intent();
        intent.objectives.push(HumanoidWholeBodyObjectiveIR::ObjectContact {
            hand: crate::morphology::HandSide::Right,
            mode: crate::whole_body_intent::HumanoidContactObjectiveMode::Acquire,
            requested_contact_force_n: 5.0,
            resulting_total_payload_kg: 0.0,
        });
        assert!(!valid_reach_intent(&intent));
    }

    #[test]
    fn spatial_invariant_is_mandatory() {
        let mut intent = reach_intent();
        intent
            .invariants
            .retain(|value| *value != HumanoidWholeBodyInvariantIR::SpatialGoalRemainsBound);
        assert!(!valid_reach_intent(&intent));
    }

    #[test]
    fn pd_gain_cardinality_and_finiteness_are_checked() {
        let gains = HumanoidPdGains::for_morphology(HumanoidMorphology::Dmc21);
        assert!(valid_pd_gains(&gains, HumanoidMorphology::Dmc21.num_actuators()));
        assert!(!valid_pd_gains(
            &gains,
            HumanoidMorphology::Dexterous53.num_actuators()
        ));
    }
}
