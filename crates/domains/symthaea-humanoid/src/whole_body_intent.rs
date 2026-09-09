// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Permit-bound whole-body motion intent (EmbodiedIR) for humanoid execution.
//!
//! This module is intentionally above joint control. It compiles one freshly
//! admitted operational skill into one atomic set of simultaneous physical
//! objectives and invariants. It never blends independently generated task
//! commands and never produces actuator values.
//!
//! The current deterministic whole-body controller only natively lowers a subset
//! of these objectives. Unsupported objectives must therefore be rejected by a
//! later lowering boundary rather than silently discarded.

use crate::morphology::{HandSide, HumanoidMorphology};
use crate::skill_permit::HumanoidSkillExecutionPermit;
use crate::skill_runtime::{
    HumanoidLocomotionMode, HumanoidSkillIntent, HumanoidSkillRequirementRole,
};
use crate::spatial_goal::HumanoidSpatiallyBoundSkillPermit;
use crate::types::ActuationMode;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum HumanoidContactObjectiveMode {
    Acquire,
    Maintain,
}

#[derive(Debug, Clone, PartialEq)]
pub enum HumanoidWholeBodyObjectiveIR {
    /// Maintain a nominal upright whole-body posture while higher-level
    /// locomotion/manipulation objectives are active.
    UprightPosture,
    /// Velocity-level locomotion demand. This is not a joint trajectory.
    LocomotionVelocity {
        mode: HumanoidLocomotionMode,
        horizontal_speed_mps: f64,
        turn_rate_rad_s: f64,
    },
    /// Body/root-frame hand target produced only from an admitted spatial goal.
    EndEffectorTarget {
        hand: HandSide,
        target_root_m: [f64; 3],
        maximum_speed_mps: f64,
    },
    /// Requested object interaction force and resulting payload state.
    ObjectContact {
        hand: HandSide,
        mode: HumanoidContactObjectiveMode,
        requested_contact_force_n: f64,
        resulting_total_payload_kg: f64,
    },
    /// Intentional human contact. Consent remains a continuously revalidated
    /// invariant and is not inferred by this objective.
    HumanContact {
        hand: HandSide,
        target_root_m: [f64; 3],
        requested_contact_force_n: f64,
        maximum_end_effector_speed_mps: f64,
    },
    /// Explicit payload carried by the body while other objectives execute.
    Payload {
        total_payload_kg: f64,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidWholeBodyInvariantIR {
    CapabilityEnvelopeRemainsAdmitted,
    QualificationSubjectRemainsStable,
    ProtectiveBehaviorMayPreemptGoal,
    SpatialGoalRemainsBound,
    LoadRetentionMaintained,
    HumanContactConsentMaintained,
}

/// Runtime description only; this struct is not authority and is deliberately
/// not Serialize. A future motor boundary must consume a fresh permit alongside
/// the intent rather than accepting a replayed intent object by itself.
#[derive(Debug, PartialEq)]
pub struct HumanoidWholeBodyMotionIntent {
    pub validation_epoch: u64,
    pub morphology: HumanoidMorphology,
    pub actuation_mode: ActuationMode,
    pub backend_profile_id: String,
    pub source_skill: HumanoidSkillIntent,
    pub requirement_subject_fingerprints: Vec<u64>,
    pub spatial_goal_id: Option<String>,
    pub objectives: Vec<HumanoidWholeBodyObjectiveIR>,
    pub invariants: Vec<HumanoidWholeBodyInvariantIR>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidWholeBodyIntentCompileError {
    SpatialBindingRequired,
    UnexpectedSpatialBinding,
    SpatialRoleMismatch,
    InvalidNumericDemand,
    MissingRequiredSemanticRole,
}

/// Compile skills that require no external spatial target.
pub fn compile_semantic_whole_body_intent(
    permit: &HumanoidSkillExecutionPermit<'_>,
) -> Result<HumanoidWholeBodyMotionIntent, HumanoidWholeBodyIntentCompileError> {
    let base_invariants = base_invariants();
    let mut objectives = vec![HumanoidWholeBodyObjectiveIR::UprightPosture];

    match permit.intent() {
        HumanoidSkillIntent::Stand => {}
        HumanoidSkillIntent::Locomote {
            mode,
            horizontal_speed_mps,
            turn_rate_rad_s,
        } => {
            if !finite_non_negative(&[horizontal_speed_mps, turn_rate_rad_s]) {
                return Err(HumanoidWholeBodyIntentCompileError::InvalidNumericDemand);
            }
            objectives.push(HumanoidWholeBodyObjectiveIR::LocomotionVelocity {
                mode,
                horizontal_speed_mps,
                turn_rate_rad_s,
            });
        }
        HumanoidSkillIntent::Reach { .. }
        | HumanoidSkillIntent::Grasp { .. }
        | HumanoidSkillIntent::Carry { .. }
        | HumanoidSkillIntent::AssistHuman { .. } => {
            return Err(HumanoidWholeBodyIntentCompileError::SpatialBindingRequired);
        }
    }

    Ok(build_intent(permit, None, objectives, base_invariants))
}

/// Compile a spatially bound Reach/Grasp/Carry/AssistHuman permit into one atomic
/// objective set. Composite skills remain simultaneous here.
pub fn compile_spatial_whole_body_intent(
    bound: &HumanoidSpatiallyBoundSkillPermit<'_>,
) -> Result<HumanoidWholeBodyMotionIntent, HumanoidWholeBodyIntentCompileError> {
    let permit = bound.semantic();
    let goal = bound.goal();
    let target_root_m = bound.target_root_m();
    let mut objectives = vec![HumanoidWholeBodyObjectiveIR::UprightPosture];
    let mut invariants = base_invariants();
    invariants.push(HumanoidWholeBodyInvariantIR::SpatialGoalRemainsBound);

    match permit.intent() {
        HumanoidSkillIntent::Stand | HumanoidSkillIntent::Locomote { .. } => {
            return Err(HumanoidWholeBodyIntentCompileError::UnexpectedSpatialBinding);
        }
        HumanoidSkillIntent::Reach {
            end_effector_speed_mps,
        } => {
            require_role(bound, HumanoidSkillRequirementRole::Manipulation)?;
            if !finite_non_negative(&[end_effector_speed_mps]) {
                return Err(HumanoidWholeBodyIntentCompileError::InvalidNumericDemand);
            }
            objectives.push(HumanoidWholeBodyObjectiveIR::EndEffectorTarget {
                hand: goal.hand,
                target_root_m,
                maximum_speed_mps: end_effector_speed_mps,
            });
        }
        HumanoidSkillIntent::Grasp {
            end_effector_speed_mps,
            object_contact_force_n,
            resulting_total_payload_kg,
        } => {
            require_role(bound, HumanoidSkillRequirementRole::Manipulation)?;
            if !finite_non_negative(&[
                end_effector_speed_mps,
                object_contact_force_n,
                resulting_total_payload_kg,
            ]) {
                return Err(HumanoidWholeBodyIntentCompileError::InvalidNumericDemand);
            }
            objectives.push(HumanoidWholeBodyObjectiveIR::EndEffectorTarget {
                hand: goal.hand,
                target_root_m,
                maximum_speed_mps: end_effector_speed_mps,
            });
            objectives.push(HumanoidWholeBodyObjectiveIR::ObjectContact {
                hand: goal.hand,
                mode: HumanoidContactObjectiveMode::Acquire,
                requested_contact_force_n: object_contact_force_n,
                resulting_total_payload_kg,
            });
            if resulting_total_payload_kg > 0.0 {
                objectives.push(HumanoidWholeBodyObjectiveIR::Payload {
                    total_payload_kg: resulting_total_payload_kg,
                });
            }
        }
        HumanoidSkillIntent::Carry {
            mode,
            horizontal_speed_mps,
            turn_rate_rad_s,
            manipulation_speed_mps,
            retention_force_n,
            resulting_total_payload_kg,
        } => {
            require_role(bound, HumanoidSkillRequirementRole::Manipulation)?;
            if !permit
                .requirements()
                .iter()
                .any(|requirement| requirement.role == HumanoidSkillRequirementRole::Locomotion)
            {
                return Err(HumanoidWholeBodyIntentCompileError::MissingRequiredSemanticRole);
            }
            if !finite_non_negative(&[
                horizontal_speed_mps,
                turn_rate_rad_s,
                manipulation_speed_mps,
                retention_force_n,
                resulting_total_payload_kg,
            ]) || resulting_total_payload_kg <= 0.0
            {
                return Err(HumanoidWholeBodyIntentCompileError::InvalidNumericDemand);
            }

            // One atomic objective graph: locomotion and manipulation are not
            // separately synthesized and blended downstream.
            objectives.push(HumanoidWholeBodyObjectiveIR::LocomotionVelocity {
                mode,
                horizontal_speed_mps,
                turn_rate_rad_s,
            });
            objectives.push(HumanoidWholeBodyObjectiveIR::EndEffectorTarget {
                hand: goal.hand,
                target_root_m,
                maximum_speed_mps: manipulation_speed_mps,
            });
            objectives.push(HumanoidWholeBodyObjectiveIR::ObjectContact {
                hand: goal.hand,
                mode: HumanoidContactObjectiveMode::Maintain,
                requested_contact_force_n: retention_force_n,
                resulting_total_payload_kg,
            });
            objectives.push(HumanoidWholeBodyObjectiveIR::Payload {
                total_payload_kg: resulting_total_payload_kg,
            });
            invariants.push(HumanoidWholeBodyInvariantIR::LoadRetentionMaintained);
        }
        HumanoidSkillIntent::AssistHuman {
            end_effector_speed_mps,
            human_contact_force_n,
        } => {
            require_role(bound, HumanoidSkillRequirementRole::HumanInteraction)?;
            if !finite_non_negative(&[end_effector_speed_mps, human_contact_force_n]) {
                return Err(HumanoidWholeBodyIntentCompileError::InvalidNumericDemand);
            }
            objectives.push(HumanoidWholeBodyObjectiveIR::HumanContact {
                hand: goal.hand,
                target_root_m,
                requested_contact_force_n: human_contact_force_n,
                maximum_end_effector_speed_mps: end_effector_speed_mps,
            });
            invariants.push(HumanoidWholeBodyInvariantIR::HumanContactConsentMaintained);
        }
    }

    Ok(build_intent(
        permit,
        Some(goal.goal_id.clone()),
        objectives,
        invariants,
    ))
}

fn build_intent(
    permit: &HumanoidSkillExecutionPermit<'_>,
    spatial_goal_id: Option<String>,
    objectives: Vec<HumanoidWholeBodyObjectiveIR>,
    invariants: Vec<HumanoidWholeBodyInvariantIR>,
) -> HumanoidWholeBodyMotionIntent {
    HumanoidWholeBodyMotionIntent {
        validation_epoch: permit.epoch(),
        morphology: permit.morphology(),
        actuation_mode: permit.actuation_mode(),
        backend_profile_id: permit.backend_profile_id().to_string(),
        source_skill: permit.intent(),
        requirement_subject_fingerprints: permit
            .requirements()
            .iter()
            .map(|requirement| requirement.request.subject_fingerprint)
            .collect(),
        spatial_goal_id,
        objectives,
        invariants,
    }
}

fn base_invariants() -> Vec<HumanoidWholeBodyInvariantIR> {
    vec![
        HumanoidWholeBodyInvariantIR::CapabilityEnvelopeRemainsAdmitted,
        HumanoidWholeBodyInvariantIR::QualificationSubjectRemainsStable,
        HumanoidWholeBodyInvariantIR::ProtectiveBehaviorMayPreemptGoal,
    ]
}

fn require_role(
    bound: &HumanoidSpatiallyBoundSkillPermit<'_>,
    expected: HumanoidSkillRequirementRole,
) -> Result<(), HumanoidWholeBodyIntentCompileError> {
    if bound.role() == expected {
        Ok(())
    } else {
        Err(HumanoidWholeBodyIntentCompileError::SpatialRoleMismatch)
    }
}

fn finite_non_negative(values: &[f64]) -> bool {
    values.iter().all(|value| value.is_finite() && *value >= 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::actuator_controllability::{
        ContactWrenchAxis, ContactWrenchAxisMargin, ContactWrenchMarginAssessment,
        HumanoidActuationControllabilityAssessment,
    };
    use crate::capability_envelope::{
        HumanInteractionEvidence, HumanoidCapabilityRestriction,
        HumanoidNominalCapabilityProfile, derive_humanoid_capability_envelope,
    };
    use crate::contact_site::HumanoidContactSite;
    use crate::execution::HumanoidAuthorityEnvelope;
    use crate::full_dynamics::DynamicsComponentSource;
    use crate::qualification::HumanoidQualificationSubject;
    use crate::skill_actuation_guard::{
        HumanoidSkillActuationEvidenceEntry, HumanoidSkillActuationPolicyEntry,
    };
    use crate::skill_executive::HumanoidSkillRuntimeEvidence;
    use crate::skill_permit::HumanoidPermitSkillExecutive;
    use crate::skill_runtime::{
        HumanoidSkillQualificationSet, compile_humanoid_skill_contract,
    };
    use crate::spatial_goal::{
        HumanoidReachWorkspaceProfile, HumanoidSpatialGoalAdmissionConfig,
        HumanoidSpatialGoalEvidence, HumanoidSpatialTargetKind, bind_humanoid_spatial_goal,
    };
    use crate::typed_actuation_capability::{
        HUMANOID_TYPED_ACTUATION_POLICY_SCHEMA_VERSION,
        TypedHumanoidActuationCapabilityPolicy, TypedHumanoidContactActuationRequirement,
    };
    use crate::types::{HumanoidState, HumanoidTask};

    fn subject(task: HumanoidTask) -> HumanoidQualificationSubject {
        HumanoidQualificationSubject::new(
            HumanoidMorphology::Dexterous53,
            task,
            ActuationMode::NormalizedTorque,
            "whole-body-intent-test-v1",
        )
    }

    fn profile(task: HumanoidTask) -> HumanoidNominalCapabilityProfile {
        HumanoidNominalCapabilityProfile::new(
            &subject(task), 2.0, 0.5, 1.0, 1.0, 0.25, 12.0, 80.0, 20.0, 0.8, true,
        )
    }

    fn envelope(task: HumanoidTask) -> crate::capability_envelope::HumanoidCapabilityEnvelope {
        derive_humanoid_capability_envelope(
            &subject(task),
            &profile(task),
            HumanoidAuthorityEnvelope::fully_admitted(),
            HumanInteractionEvidence::no_human_present(),
        )
    }

    fn policy(
        task: HumanoidTask,
        role: HumanoidSkillRequirementRole,
        site: HumanoidContactSite,
    ) -> HumanoidSkillActuationPolicyEntry {
        let subject = subject(task);
        HumanoidSkillActuationPolicyEntry {
            role,
            subject: subject.clone(),
            profile: profile(task),
            policy: TypedHumanoidActuationCapabilityPolicy {
                schema_version: HUMANOID_TYPED_ACTUATION_POLICY_SCHEMA_VERSION,
                subject_fingerprint: subject.fingerprint(),
                authority_profile_id: "whole-body-joints-v1".into(),
                calibration_fingerprint: 11,
                dynamics_model_id: "whole-body-dynamics-v1".into(),
                requirements: vec![TypedHumanoidContactActuationRequirement {
                    site,
                    axis: ContactWrenchAxis::ForceZ,
                    nominal_min_retained_fraction: 0.8,
                    degraded_min_retained_fraction: 0.5,
                    nominal_min_retained_wrench: 20.0,
                    degraded_min_retained_wrench: 10.0,
                }],
                degraded_restriction: HumanoidCapabilityRestriction {
                    max_horizontal_speed_mps: 0.5,
                    max_turn_rate_rad_s: 0.3,
                    max_end_effector_speed_mps: 0.3,
                    max_payload_kg: 5.0,
                    max_object_contact_force_n: 40.0,
                    max_human_contact_force_n: 10.0,
                },
            },
        }
    }

    fn evidence(
        task: HumanoidTask,
        site: HumanoidContactSite,
    ) -> HumanoidSkillActuationEvidenceEntry {
        HumanoidSkillActuationEvidenceEntry {
            subject_fingerprint: subject(task).fingerprint(),
            assessment: HumanoidActuationControllabilityAssessment {
                morphology: HumanoidMorphology::Dexterous53,
                authority_sequence: 1,
                authority_age_s: 0.01,
                authority_profile_id: "whole-body-joints-v1".into(),
                calibration_fingerprint: 11,
                dynamics_model_id: "whole-body-dynamics-v1".into(),
                sites: vec![ContactWrenchMarginAssessment {
                    site_id: site.canonical_id().into(),
                    contact_confidence: 1.0,
                    jacobian_source: DynamicsComponentSource::SimulatorSolver,
                    actuator_limit_source: DynamicsComponentSource::SystemIdentification,
                    axes: vec![ContactWrenchAxisMargin {
                        axis: ContactWrenchAxis::ForceZ,
                        actuated_support_present: true,
                        nominal_limit: Some(30.0),
                        retained_limit: Some(27.0),
                        retained_fraction: 0.9,
                        limiting_joint: Some(0),
                    }],
                    minimum_retained_fraction: 0.9,
                }],
            },
        }
    }

    fn runtime(load_retained: bool) -> HumanoidSkillRuntimeEvidence {
        HumanoidSkillRuntimeEvidence {
            goal_authority_valid: true,
            load_retained,
            human_proximity_valid: true,
            human_contact_consent: false,
            protective_preempted: false,
            objective_satisfied: false,
        }
    }

    fn state() -> HumanoidState {
        let mut state = HumanoidState::standing_for(HumanoidMorphology::Dexterous53);
        state.root_position = [0.0, 0.0, 1.0];
        state.root_quaternion = [1.0, 0.0, 0.0, 0.0];
        state.timestamp = 1.0;
        state
    }

    #[test]
    fn plain_locomotion_compiles_without_spatial_binding() {
        let qualifications = HumanoidSkillQualificationSet::new(vec![subject(HumanoidTask::Walk)]);
        let contract = compile_humanoid_skill_contract(
            HumanoidSkillIntent::Locomote {
                mode: HumanoidLocomotionMode::Walk,
                horizontal_speed_mps: 0.4,
                turn_rate_rad_s: 0.1,
            },
            &qualifications,
        )
        .unwrap();
        let mut executive = HumanoidPermitSkillExecutive::new();
        let cycle = executive
            .start_and_issue(
                contract,
                &[envelope(HumanoidTask::Walk)],
                &[policy(
                    HumanoidTask::Walk,
                    HumanoidSkillRequirementRole::Locomotion,
                    HumanoidContactSite::RightFoot,
                )],
                &[evidence(HumanoidTask::Walk, HumanoidContactSite::RightFoot)],
                runtime(false),
            )
            .unwrap();
        let permit = cycle.skill_permit();
        let intent = compile_semantic_whole_body_intent(&permit).unwrap();
        assert!(intent.objectives.iter().any(|objective| matches!(
            objective,
            HumanoidWholeBodyObjectiveIR::LocomotionVelocity { .. }
        )));
    }

    #[test]
    fn carry_compiles_as_one_simultaneous_locomotion_and_manipulation_graph() {
        let qualifications = HumanoidSkillQualificationSet::new(vec![
            subject(HumanoidTask::Walk),
            subject(HumanoidTask::Grasp),
        ]);
        let contract = compile_humanoid_skill_contract(
            HumanoidSkillIntent::Carry {
                mode: HumanoidLocomotionMode::Walk,
                horizontal_speed_mps: 0.4,
                turn_rate_rad_s: 0.1,
                manipulation_speed_mps: 0.1,
                retention_force_n: 10.0,
                resulting_total_payload_kg: 3.0,
            },
            &qualifications,
        )
        .unwrap();
        let policies = vec![
            policy(
                HumanoidTask::Grasp,
                HumanoidSkillRequirementRole::Manipulation,
                HumanoidContactSite::RightHand,
            ),
            policy(
                HumanoidTask::Walk,
                HumanoidSkillRequirementRole::Locomotion,
                HumanoidContactSite::RightFoot,
            ),
        ];
        let evidence = vec![
            evidence(HumanoidTask::Grasp, HumanoidContactSite::RightHand),
            evidence(HumanoidTask::Walk, HumanoidContactSite::RightFoot),
        ];
        let envelopes = [envelope(HumanoidTask::Grasp), envelope(HumanoidTask::Walk)];
        let mut executive = HumanoidPermitSkillExecutive::new();
        let cycle = executive
            .start_and_issue(contract, &envelopes, &policies, &evidence, runtime(true))
            .unwrap();
        let workspace = HumanoidReachWorkspaceProfile::from_subject(
            &subject(HumanoidTask::Grasp),
            HandSide::Right,
            [0.3, -0.2, 0.2],
            [0.5, 0.5, 0.5],
        );
        let goal = HumanoidSpatialGoalEvidence {
            goal_id: "carried-object".into(),
            kind: HumanoidSpatialTargetKind::Object,
            hand: HandSide::Right,
            target_world_m: [0.3, -0.2, 1.2],
            observed_at_s: 1.0,
            received_at_s: 1.0,
            confidence: 0.95,
        };
        let bound = bind_humanoid_spatial_goal(
            &cycle,
            HumanoidSkillRequirementRole::Manipulation,
            &subject(HumanoidTask::Grasp),
            &workspace,
            &goal,
            &policies[0],
            &evidence[0],
            &state(),
            1.05,
            HumanoidSpatialGoalAdmissionConfig {
                maximum_goal_age_s: 0.2,
                maximum_state_age_s: 0.2,
                minimum_goal_confidence: 0.8,
            },
        )
        .unwrap();
        let intent = compile_spatial_whole_body_intent(&bound).unwrap();

        assert!(intent.objectives.iter().any(|objective| matches!(
            objective,
            HumanoidWholeBodyObjectiveIR::LocomotionVelocity { .. }
        )));
        assert!(intent.objectives.iter().any(|objective| matches!(
            objective,
            HumanoidWholeBodyObjectiveIR::EndEffectorTarget { .. }
        )));
        assert!(intent.objectives.iter().any(|objective| matches!(
            objective,
            HumanoidWholeBodyObjectiveIR::ObjectContact {
                mode: HumanoidContactObjectiveMode::Maintain,
                ..
            }
        )));
        assert!(intent.objectives.iter().any(|objective| matches!(
            objective,
            HumanoidWholeBodyObjectiveIR::Payload { total_payload_kg } if *total_payload_kg == 3.0
        )));
        assert!(intent
            .invariants
            .contains(&HumanoidWholeBodyInvariantIR::LoadRetentionMaintained));
    }

    #[test]
    fn spatial_skill_cannot_compile_from_semantic_permit_alone() {
        let qualifications = HumanoidSkillQualificationSet::new(vec![subject(HumanoidTask::Reach)]);
        let contract = compile_humanoid_skill_contract(
            HumanoidSkillIntent::Reach {
                end_effector_speed_mps: 0.1,
            },
            &qualifications,
        )
        .unwrap();
        let mut executive = HumanoidPermitSkillExecutive::new();
        let cycle = executive
            .start_and_issue(
                contract,
                &[envelope(HumanoidTask::Reach)],
                &[policy(
                    HumanoidTask::Reach,
                    HumanoidSkillRequirementRole::Manipulation,
                    HumanoidContactSite::RightHand,
                )],
                &[evidence(HumanoidTask::Reach, HumanoidContactSite::RightHand)],
                runtime(false),
            )
            .unwrap();
        let permit = cycle.skill_permit();
        assert_eq!(
            compile_semantic_whole_body_intent(&permit),
            Err(HumanoidWholeBodyIntentCompileError::SpatialBindingRequired)
        );
    }
}
