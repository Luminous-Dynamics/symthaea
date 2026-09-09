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
use crate::execution::{
    HumanoidAuthorityEnvelope, HumanoidExecutionPipeline, HumanoidExecutionResult,
    HumanoidPreparedCommand,
};
use crate::floating_base::FloatingBaseDynamicsProvider;
use crate::frozen_dynamics::FrozenHumanoidDynamicsEnvironment;
use crate::full_dynamics::FullRigidBodyDynamicsProvider;
use crate::skill_runtime::HumanoidSkillIntent;
use crate::spatial_goal::HumanoidSpatiallyBoundSkillPermit;
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
    InvalidPdGains,
    InvalidReachIntent,
    MissingFullDynamics,
    CartesianReference(HumanoidCartesianHandReferenceFailure),
}

/// Exact snapshot identities frozen for one preparation cycle. Optional reduced
/// and floating-base contracts remain explicit rather than being promoted to
/// evidence when the backend did not provide them.
#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidFrozenDynamicsLineage {
    pub rigid_model_id: Option<String>,
    pub rigid_sampled_at_s: Option<f64>,
    pub full_model_id: String,
    pub full_sampled_at_s: f64,
    pub floating_model_id: Option<String>,
    pub floating_sampled_at_s: Option<f64>,
}

/// Non-authoritative evidence emitted when one Reach command has been prepared.
#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidPermittedReachPreparationReport {
    pub validation_epoch: u64,
    pub goal_id: String,
    pub dynamics: HumanoidFrozenDynamicsLineage,
    pub cartesian_reference: HumanoidCartesianHandReferenceReport,
}

/// Final actuator-eligible result after the prepared Reach command crosses typed
/// goal authority and the pipeline's final safety projector.
#[derive(Debug, Clone)]
pub struct HumanoidPermittedReachExecutionResult {
    pub execution: HumanoidExecutionResult,
    pub preparation: HumanoidPermittedReachPreparationReport,
}

/// One-shot Reach command that is still below the final authority/safety
/// boundary.
///
/// This type is deliberately not Clone or Serialize. Holding the spatial permit
/// keeps the originating validation cycle borrowed, while holding the mutable
/// pipeline borrow pins the command to the exact safety-projector history used
/// for preparation.
pub struct HumanoidPermittedReachPreparedCommand<'pipeline, 'permit> {
    pipeline: &'pipeline mut HumanoidExecutionPipeline,
    #[allow(dead_code)]
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

    /// Consume the one-shot prepared command and cross typed authority plus the
    /// exact pipeline instance's final morphology-aware safety projection.
    pub fn finalize(
        self,
        authority: HumanoidAuthorityEnvelope,
    ) -> HumanoidPermittedReachExecutionResult {
        let Self {
            pipeline,
            permit: _,
            prepared,
            state,
            actuation_mode,
            dt,
            report,
        } = self;
        let execution =
            pipeline.finalize_prepared(prepared, &state, authority, actuation_mode, dt);
        HumanoidPermittedReachExecutionResult {
            execution,
            preparation: report,
        }
    }
}

/// Prepare one permit-bound Reach command through the existing deterministic
/// hierarchy.
///
/// The Cartesian hand correction is added to a standing reference *before* the
/// hierarchy runs. Sparse inverse dynamics, floating-base dynamics, contact
/// dynamics, centroidal correction, typed authority, final projection, and HAL
/// therefore remain downstream of manipulation reference generation.
///
/// Every exposed body-dynamics contract is sampled exactly once into a frozen
/// per-cycle adapter. Cartesian lowering and the existing hierarchy therefore
/// consume the same full-dynamics snapshot rather than independently sampling a
/// live backend at two different instants.
///
/// This path intentionally supports Reach only. Grasp/Carry/HumanContact remain
/// rejected until their contact-force and retention semantics are implemented in
/// the native whole-body solver.
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
    if !valid_pd_gains(pd_gains, pipeline.morphology().num_actuators()) {
        return Err(HumanoidPermittedReachPreparationFailure::InvalidPdGains);
    }
    if !valid_reach_intent(intent) {
        return Err(HumanoidPermittedReachPreparationFailure::InvalidReachIntent);
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

    // Reach reference is fully deterministic in this first execution path. No
    // learned residual is injected here; learned proposals can be added later as
    // separately bounded references without changing this authority boundary.
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
        goal_id: permit.goal().goal_id.clone(),
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
        && gains
            .kp
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0)
        && gains
            .kd
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0)
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
            // Reach execution must not erase manipulation/contact semantics from
            // a stronger skill just to make it runnable.
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
        assert!(valid_pd_gains(
            &gains,
            HumanoidMorphology::Dmc21.num_actuators()
        ));
        assert!(!valid_pd_gains(
            &gains,
            HumanoidMorphology::Dexterous53.num_actuators()
        ));
    }
}
