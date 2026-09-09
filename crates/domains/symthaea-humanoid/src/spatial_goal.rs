// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Spatial goal binding for freshly permitted humanoid skills.
//!
//! Semantic capability admission alone does not authorize an arbitrary point in
//! space. Reach/Grasp/Carry/AssistHuman must bind the selected hand and target to
//! explicit workspace qualification, fresh target evidence, fresh body state,
//! and fresh actuation evidence for that same hand before a whole-body motion
//! intent may be compiled.

use serde::{Deserialize, Serialize};

use crate::actuation_capability::HumanoidActuationCapabilityAction;
use crate::capability_envelope::HumanoidNominalCapabilityProfile;
use crate::contact_site::HumanoidContactSite;
use crate::morphology::{HandSide, HumanoidMorphology};
use crate::qualification::HumanoidQualificationSubject;
use crate::skill_actuation_guard::{
    HumanoidSkillActuationEvidenceEntry, HumanoidSkillActuationPolicyEntry,
};
use crate::skill_permit::{
    HumanoidSkillExecutionPermit, HumanoidSkillValidationCycle,
};
use crate::skill_runtime::{
    HumanoidSkillRequirementRole,
};
use crate::typed_actuation_capability::assess_typed_humanoid_actuation_capability;
use crate::types::HumanoidState;

pub const HUMANOID_REACH_WORKSPACE_PROFILE_SCHEMA_VERSION: u32 = 1;

/// Qualified root-frame workspace for one exact task/body/backend and one hand.
///
/// The workspace is an ellipsoid. Values are qualification outputs, not built-in
/// assumptions, so this type intentionally has no Default implementation.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HumanoidReachWorkspaceProfile {
    pub schema_version: u32,
    pub subject_fingerprint: u64,
    pub morphology: HumanoidMorphology,
    pub hand: HandSide,
    /// Ellipsoid center in the humanoid root/body frame, metres.
    pub center_root_m: [f64; 3],
    /// Positive ellipsoid semi-axes in metres.
    pub semi_axes_m: [f64; 3],
}

impl HumanoidReachWorkspaceProfile {
    pub fn from_subject(
        subject: &HumanoidQualificationSubject,
        hand: HandSide,
        center_root_m: [f64; 3],
        semi_axes_m: [f64; 3],
    ) -> Self {
        Self {
            schema_version: HUMANOID_REACH_WORKSPACE_PROFILE_SCHEMA_VERSION,
            subject_fingerprint: subject.fingerprint(),
            morphology: subject.morphology,
            hand,
            center_root_m,
            semi_axes_m,
        }
    }

    pub fn validate_for(&self, subject: &HumanoidQualificationSubject) -> bool {
        self.schema_version == HUMANOID_REACH_WORKSPACE_PROFILE_SCHEMA_VERSION
            && subject.validate()
            && self.subject_fingerprint != 0
            && self.subject_fingerprint == subject.fingerprint()
            && self.morphology == subject.morphology
            && self.center_root_m.iter().all(|v| v.is_finite())
            && self
                .semi_axes_m
                .iter()
                .all(|v| v.is_finite() && *v > 0.0)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HumanoidSpatialTargetKind {
    Object,
    HumanContact,
    GenericPoint,
}

/// Perception/planning evidence for one exact target selected for the skill.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HumanoidSpatialGoalEvidence {
    pub goal_id: String,
    pub kind: HumanoidSpatialTargetKind,
    pub hand: HandSide,
    pub target_world_m: [f64; 3],
    pub observed_at_s: f64,
    pub received_at_s: f64,
    /// Producer confidence. This module only applies an explicit admission floor;
    /// it does not turn confidence into a fabricated workspace size.
    pub confidence: f64,
}

impl HumanoidSpatialGoalEvidence {
    pub fn validate(&self) -> bool {
        !self.goal_id.trim().is_empty()
            && self.goal_id == self.goal_id.trim()
            && self.goal_id.len() <= 256
            && self.target_world_m.iter().all(|v| v.is_finite())
            && self.observed_at_s.is_finite()
            && self.received_at_s.is_finite()
            && self.observed_at_s >= 0.0
            && self.received_at_s >= self.observed_at_s
            && self.confidence.is_finite()
            && (0.0..=1.0).contains(&self.confidence)
    }
}

/// Explicit freshness/confidence policy. No Default: production callers must bind
/// these values to their actual perception/state-estimation qualification.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct HumanoidSpatialGoalAdmissionConfig {
    pub maximum_goal_age_s: f64,
    pub maximum_state_age_s: f64,
    pub minimum_goal_confidence: f64,
}

impl HumanoidSpatialGoalAdmissionConfig {
    pub fn validate(self) -> bool {
        self.maximum_goal_age_s.is_finite()
            && self.maximum_goal_age_s > 0.0
            && self.maximum_state_age_s.is_finite()
            && self.maximum_state_age_s > 0.0
            && self.minimum_goal_confidence.is_finite()
            && (0.0..=1.0).contains(&self.minimum_goal_confidence)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HumanoidSpatialGoalFailure {
    InvalidConfig,
    InvalidGoalEvidence,
    InvalidWorkspaceProfile,
    MissingRequirement,
    DuplicateRequirement,
    WorkspaceSubjectMismatch,
    GoalKindMismatch,
    GoalHandMismatch,
    GoalTimestampInFuture,
    GoalEvidenceStale,
    GoalConfidenceTooLow,
    InvalidState,
    StateTimestampInFuture,
    StateEvidenceStale,
    TargetOutsideWorkspace,
    ActuationPolicyMismatch,
    SelectedHandNotQualified,
    ActuationEvidenceMismatch,
    SelectedHandActuationHold,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HumanoidSpatialGoalAdmission {
    pub admitted: bool,
    pub failure: Option<HumanoidSpatialGoalFailure>,
    pub subject_fingerprint: u64,
    pub goal_id: String,
    pub target_root_m: [f64; 3],
    /// Squared normalized ellipsoid radius; <=1 is inside the qualified workspace.
    pub workspace_utilization_sq: f64,
    pub actuation_action: Option<HumanoidActuationCapabilityAction>,
}

/// Non-cloneable, non-serializable refinement of one fresh skill permit with an
/// exact spatial target and hand that passed workspace + actuation admission.
pub struct HumanoidSpatiallyBoundSkillPermit<'cycle> {
    semantic: HumanoidSkillExecutionPermit<'cycle>,
    role: HumanoidSkillRequirementRole,
    goal: HumanoidSpatialGoalEvidence,
    target_root_m: [f64; 3],
    workspace_utilization_sq: f64,
    selected_hand_actuation: HumanoidActuationCapabilityAction,
}

impl<'cycle> HumanoidSpatiallyBoundSkillPermit<'cycle> {
    pub fn semantic(&self) -> &HumanoidSkillExecutionPermit<'cycle> {
        &self.semantic
    }

    pub const fn role(&self) -> HumanoidSkillRequirementRole {
        self.role
    }

    pub fn goal(&self) -> &HumanoidSpatialGoalEvidence {
        &self.goal
    }

    pub const fn target_root_m(&self) -> [f64; 3] {
        self.target_root_m
    }

    pub const fn workspace_utilization_sq(&self) -> f64 {
        self.workspace_utilization_sq
    }

    pub const fn selected_hand_actuation(&self) -> HumanoidActuationCapabilityAction {
        self.selected_hand_actuation
    }
}

/// Bind an exact spatial target to a freshly validated skill cycle.
///
/// `role` must identify exactly one semantic requirement in the skill. For
/// ordinary Reach/Grasp/Carry this is `Manipulation`; AssistHuman uses
/// `HumanInteraction`.
#[allow(clippy::too_many_arguments)]
pub fn bind_humanoid_spatial_goal<'cycle, 'exec>(
    cycle: &'cycle HumanoidSkillValidationCycle<'exec>,
    role: HumanoidSkillRequirementRole,
    workspace_subject: &HumanoidQualificationSubject,
    workspace: &HumanoidReachWorkspaceProfile,
    goal: &HumanoidSpatialGoalEvidence,
    actuation_policy: &HumanoidSkillActuationPolicyEntry,
    actuation_evidence: &HumanoidSkillActuationEvidenceEntry,
    state: &HumanoidState,
    now_s: f64,
    config: HumanoidSpatialGoalAdmissionConfig,
) -> Result<HumanoidSpatiallyBoundSkillPermit<'cycle>, HumanoidSpatialGoalAdmission> {
    let semantic = cycle.skill_permit();
    let requirements = semantic
        .requirements()
        .iter()
        .filter(|requirement| requirement.role == role)
        .collect::<Vec<_>>();
    let subject_fingerprint = requirements
        .first()
        .map(|requirement| requirement.request.subject_fingerprint)
        .unwrap_or(0);

    let reject = |failure, target_root_m, utilization, action| {
        Err(HumanoidSpatialGoalAdmission {
            admitted: false,
            failure: Some(failure),
            subject_fingerprint,
            goal_id: goal.goal_id.clone(),
            target_root_m,
            workspace_utilization_sq: utilization,
            actuation_action: action,
        })
    };

    if !config.validate() || !now_s.is_finite() || now_s < 0.0 {
        return reject(HumanoidSpatialGoalFailure::InvalidConfig, [0.0; 3], f64::INFINITY, None);
    }
    if !goal.validate() {
        return reject(
            HumanoidSpatialGoalFailure::InvalidGoalEvidence,
            [0.0; 3],
            f64::INFINITY,
            None,
        );
    }
    let requirement = match requirements.as_slice() {
        [] => {
            return reject(
                HumanoidSpatialGoalFailure::MissingRequirement,
                [0.0; 3],
                f64::INFINITY,
                None,
            )
        }
        [requirement] => *requirement,
        _ => {
            return reject(
                HumanoidSpatialGoalFailure::DuplicateRequirement,
                [0.0; 3],
                f64::INFINITY,
                None,
            )
        }
    };

    if requirement.request.subject_fingerprint != workspace_subject.fingerprint()
        || !workspace.validate_for(workspace_subject)
        || workspace_subject.morphology != semantic.morphology()
        || workspace_subject.actuation_mode != semantic.actuation_mode()
        || workspace_subject.backend_profile_id != semantic.backend_profile_id()
    {
        return reject(
            HumanoidSpatialGoalFailure::WorkspaceSubjectMismatch,
            [0.0; 3],
            f64::INFINITY,
            None,
        );
    }
    if goal.hand != workspace.hand {
        return reject(
            HumanoidSpatialGoalFailure::GoalHandMismatch,
            [0.0; 3],
            f64::INFINITY,
            None,
        );
    }
    if role == HumanoidSkillRequirementRole::HumanInteraction
        && goal.kind != HumanoidSpatialTargetKind::HumanContact
    {
        return reject(
            HumanoidSpatialGoalFailure::GoalKindMismatch,
            [0.0; 3],
            f64::INFINITY,
            None,
        );
    }
    if goal.received_at_s > now_s {
        return reject(
            HumanoidSpatialGoalFailure::GoalTimestampInFuture,
            [0.0; 3],
            f64::INFINITY,
            None,
        );
    }
    if now_s - goal.observed_at_s > config.maximum_goal_age_s {
        return reject(
            HumanoidSpatialGoalFailure::GoalEvidenceStale,
            [0.0; 3],
            f64::INFINITY,
            None,
        );
    }
    if goal.confidence < config.minimum_goal_confidence {
        return reject(
            HumanoidSpatialGoalFailure::GoalConfidenceTooLow,
            [0.0; 3],
            f64::INFINITY,
            None,
        );
    }

    let Some(target_root_m) = world_point_in_root_frame(state, goal.target_world_m) else {
        return reject(
            HumanoidSpatialGoalFailure::InvalidState,
            [0.0; 3],
            f64::INFINITY,
            None,
        );
    };
    if state.timestamp > now_s {
        return reject(
            HumanoidSpatialGoalFailure::StateTimestampInFuture,
            target_root_m,
            f64::INFINITY,
            None,
        );
    }
    if now_s - state.timestamp > config.maximum_state_age_s {
        return reject(
            HumanoidSpatialGoalFailure::StateEvidenceStale,
            target_root_m,
            f64::INFINITY,
            None,
        );
    }

    let utilization = workspace_utilization_sq(workspace, target_root_m);
    if !utilization.is_finite() || utilization > 1.0 {
        return reject(
            HumanoidSpatialGoalFailure::TargetOutsideWorkspace,
            target_root_m,
            utilization,
            None,
        );
    }

    if actuation_policy.role != role
        || actuation_policy.subject.fingerprint() != subject_fingerprint
        || !actuation_policy.validate()
    {
        return reject(
            HumanoidSpatialGoalFailure::ActuationPolicyMismatch,
            target_root_m,
            utilization,
            None,
        );
    }
    let selected_hand_site = hand_site(goal.hand);
    if !actuation_policy
        .policy
        .requirements
        .iter()
        .any(|requirement| requirement.site == selected_hand_site)
    {
        return reject(
            HumanoidSpatialGoalFailure::SelectedHandNotQualified,
            target_root_m,
            utilization,
            None,
        );
    }
    if actuation_evidence.subject_fingerprint != subject_fingerprint
        || actuation_evidence.assessment.morphology != workspace_subject.morphology
    {
        return reject(
            HumanoidSpatialGoalFailure::ActuationEvidenceMismatch,
            target_root_m,
            utilization,
            None,
        );
    }

    let actuation_decision = assess_typed_humanoid_actuation_capability(
        &actuation_policy.subject,
        &actuation_policy.profile,
        &actuation_policy.policy,
        &actuation_evidence.assessment,
    );
    if actuation_decision.action == HumanoidActuationCapabilityAction::Hold {
        return reject(
            HumanoidSpatialGoalFailure::SelectedHandActuationHold,
            target_root_m,
            utilization,
            Some(actuation_decision.action),
        );
    }

    Ok(HumanoidSpatiallyBoundSkillPermit {
        semantic,
        role,
        goal: goal.clone(),
        target_root_m,
        workspace_utilization_sq: utilization,
        selected_hand_actuation: actuation_decision.action,
    })
}

fn hand_site(hand: HandSide) -> HumanoidContactSite {
    match hand {
        HandSide::Right => HumanoidContactSite::RightHand,
        HandSide::Left => HumanoidContactSite::LeftHand,
    }
}

fn workspace_utilization_sq(
    profile: &HumanoidReachWorkspaceProfile,
    target_root_m: [f64; 3],
) -> f64 {
    (0..3)
        .map(|axis| {
            let normalized =
                (target_root_m[axis] - profile.center_root_m[axis]) / profile.semi_axes_m[axis];
            normalized * normalized
        })
        .sum()
}

/// Transform one world-space point into the humanoid root/body frame.
fn world_point_in_root_frame(state: &HumanoidState, point_world: [f64; 3]) -> Option<[f64; 3]> {
    if !state.root_position.iter().all(|v| v.is_finite())
        || !state.root_quaternion.iter().all(|v| v.is_finite())
        || !state.timestamp.is_finite()
        || state.timestamp < 0.0
    {
        return None;
    }
    let q = normalized_quaternion(state.root_quaternion)?;
    let relative = [
        point_world[0] - state.root_position[0],
        point_world[1] - state.root_position[1],
        point_world[2] - state.root_position[2],
    ];
    // Root quaternion maps body -> world. Its conjugate maps world -> body.
    Some(rotate_vector([q[0], -q[1], -q[2], -q[3]], relative))
}

fn normalized_quaternion(q: [f64; 4]) -> Option<[f64; 4]> {
    let norm_sq = q.iter().map(|value| value * value).sum::<f64>();
    if !norm_sq.is_finite() || norm_sq <= 1e-12 {
        return None;
    }
    let inv = norm_sq.sqrt().recip();
    Some([q[0] * inv, q[1] * inv, q[2] * inv, q[3] * inv])
}

fn rotate_vector(q: [f64; 4], v: [f64; 3]) -> [f64; 3] {
    let [w, x, y, z] = q;
    let qv = [x, y, z];
    let uv = cross(qv, v);
    let uuv = cross(qv, uv);
    [
        v[0] + 2.0 * (w * uv[0] + uuv[0]),
        v[1] + 2.0 * (w * uv[1] + uuv[1]),
        v[2] + 2.0 * (w * uv[2] + uuv[2]),
    ]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
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
        derive_humanoid_capability_envelope,
    };
    use crate::execution::HumanoidAuthorityEnvelope;
    use crate::full_dynamics::DynamicsComponentSource;
    use crate::skill_permit::HumanoidPermitSkillExecutive;
    use crate::skill_runtime::{
        HumanoidSkillIntent, HumanoidSkillQualificationSet,
        compile_humanoid_skill_contract,
    };
    use crate::typed_actuation_capability::{
        HUMANOID_TYPED_ACTUATION_POLICY_SCHEMA_VERSION,
        TypedHumanoidActuationCapabilityPolicy, TypedHumanoidContactActuationRequirement,
    };
    use crate::types::{ActuationMode, HumanoidTask};

    fn subject() -> HumanoidQualificationSubject {
        HumanoidQualificationSubject::new(
            HumanoidMorphology::Dexterous53,
            HumanoidTask::Reach,
            ActuationMode::NormalizedTorque,
            "spatial-test-backend-v1",
        )
    }

    fn capability_profile() -> HumanoidNominalCapabilityProfile {
        HumanoidNominalCapabilityProfile::new(
            &subject(), 0.0, 0.0, 0.0, 1.0, 0.2, 5.0, 40.0, 10.0, 0.8, true,
        )
    }

    fn envelope() -> crate::capability_envelope::HumanoidCapabilityEnvelope {
        derive_humanoid_capability_envelope(
            &subject(),
            &capability_profile(),
            HumanoidAuthorityEnvelope::fully_admitted(),
            HumanInteractionEvidence::no_human_present(),
        )
    }

    fn policy(hand: HandSide) -> HumanoidSkillActuationPolicyEntry {
        let subject = subject();
        HumanoidSkillActuationPolicyEntry {
            role: HumanoidSkillRequirementRole::Manipulation,
            subject: subject.clone(),
            profile: capability_profile(),
            policy: TypedHumanoidActuationCapabilityPolicy {
                schema_version: HUMANOID_TYPED_ACTUATION_POLICY_SCHEMA_VERSION,
                subject_fingerprint: subject.fingerprint(),
                authority_profile_id: "spatial-joints-v1".into(),
                calibration_fingerprint: 9,
                dynamics_model_id: "spatial-dynamics-v1".into(),
                requirements: vec![TypedHumanoidContactActuationRequirement {
                    site: hand_site(hand),
                    axis: ContactWrenchAxis::ForceZ,
                    nominal_min_retained_fraction: 0.8,
                    degraded_min_retained_fraction: 0.5,
                    nominal_min_retained_wrench: 20.0,
                    degraded_min_retained_wrench: 10.0,
                }],
                degraded_restriction: HumanoidCapabilityRestriction {
                    max_horizontal_speed_mps: 0.0,
                    max_turn_rate_rad_s: 0.0,
                    max_end_effector_speed_mps: 0.3,
                    max_payload_kg: 3.0,
                    max_object_contact_force_n: 20.0,
                    max_human_contact_force_n: 5.0,
                },
            },
        }
    }

    fn evidence(hand: HandSide) -> HumanoidSkillActuationEvidenceEntry {
        HumanoidSkillActuationEvidenceEntry {
            subject_fingerprint: subject().fingerprint(),
            assessment: HumanoidActuationControllabilityAssessment {
                morphology: HumanoidMorphology::Dexterous53,
                authority_sequence: 1,
                authority_age_s: 0.01,
                authority_profile_id: "spatial-joints-v1".into(),
                calibration_fingerprint: 9,
                dynamics_model_id: "spatial-dynamics-v1".into(),
                sites: vec![ContactWrenchMarginAssessment {
                    site_id: hand_site(hand).canonical_id().into(),
                    contact_confidence: 1.0,
                    jacobian_source: DynamicsComponentSource::SimulatorSolver,
                    actuator_limit_source: DynamicsComponentSource::SystemIdentification,
                    axes: vec![ContactWrenchAxisMargin {
                        axis: ContactWrenchAxis::ForceZ,
                        actuated_support_present: true,
                        nominal_limit: Some(30.0),
                        retained_limit: Some(27.0),
                        retained_fraction: 0.9,
                        limiting_joint: Some(15),
                    }],
                    minimum_retained_fraction: 0.9,
                }],
            },
        }
    }

    fn cycle<'a>(executive: &'a mut HumanoidPermitSkillExecutive) -> HumanoidSkillValidationCycle<'a> {
        let qualifications = HumanoidSkillQualificationSet::new(vec![subject()]);
        let contract = compile_humanoid_skill_contract(
            HumanoidSkillIntent::Reach {
                end_effector_speed_mps: 0.1,
            },
            &qualifications,
        )
        .unwrap();
        executive
            .start_and_issue(
                contract,
                &[envelope()],
                &[policy(HandSide::Right)],
                &[evidence(HandSide::Right)],
                crate::skill_executive::HumanoidSkillRuntimeEvidence {
                    goal_authority_valid: true,
                    load_retained: false,
                    human_proximity_valid: true,
                    human_contact_consent: false,
                    protective_preempted: false,
                    objective_satisfied: false,
                },
            )
            .unwrap()
    }

    fn state() -> HumanoidState {
        let mut state = HumanoidState::standing_for(HumanoidMorphology::Dexterous53);
        state.root_position = [1.0, 2.0, 1.0];
        state.root_quaternion = [1.0, 0.0, 0.0, 0.0];
        state.timestamp = 10.0;
        state
    }

    fn workspace() -> HumanoidReachWorkspaceProfile {
        HumanoidReachWorkspaceProfile::from_subject(
            &subject(),
            HandSide::Right,
            [0.3, -0.2, 0.2],
            [0.4, 0.4, 0.4],
        )
    }

    fn goal(hand: HandSide, target_world_m: [f64; 3]) -> HumanoidSpatialGoalEvidence {
        HumanoidSpatialGoalEvidence {
            goal_id: "object-17".into(),
            kind: HumanoidSpatialTargetKind::Object,
            hand,
            target_world_m,
            observed_at_s: 10.0,
            received_at_s: 10.0,
            confidence: 0.95,
        }
    }

    fn config() -> HumanoidSpatialGoalAdmissionConfig {
        HumanoidSpatialGoalAdmissionConfig {
            maximum_goal_age_s: 0.2,
            maximum_state_age_s: 0.2,
            minimum_goal_confidence: 0.8,
        }
    }

    #[test]
    fn qualified_right_hand_target_binds_inside_workspace() {
        let mut executive = HumanoidPermitSkillExecutive::new();
        let cycle = cycle(&mut executive);
        let bound = bind_humanoid_spatial_goal(
            &cycle,
            HumanoidSkillRequirementRole::Manipulation,
            &subject(),
            &workspace(),
            &goal(HandSide::Right, [1.3, 1.8, 1.2]),
            &policy(HandSide::Right),
            &evidence(HandSide::Right),
            &state(),
            10.05,
            config(),
        )
        .unwrap();
        assert!(bound.workspace_utilization_sq() <= 1.0);
        assert_eq!(bound.goal().goal_id, "object-17");
        assert_eq!(bound.selected_hand_actuation(), HumanoidActuationCapabilityAction::Continue);
    }

    #[test]
    fn target_outside_qualified_workspace_is_rejected() {
        let mut executive = HumanoidPermitSkillExecutive::new();
        let cycle = cycle(&mut executive);
        let rejection = bind_humanoid_spatial_goal(
            &cycle,
            HumanoidSkillRequirementRole::Manipulation,
            &subject(),
            &workspace(),
            &goal(HandSide::Right, [3.0, 2.0, 1.0]),
            &policy(HandSide::Right),
            &evidence(HandSide::Right),
            &state(),
            10.05,
            config(),
        )
        .unwrap_err();
        assert_eq!(rejection.failure, Some(HumanoidSpatialGoalFailure::TargetOutsideWorkspace));
    }

    #[test]
    fn selected_hand_must_be_explicitly_actuation_qualified() {
        let mut executive = HumanoidPermitSkillExecutive::new();
        let cycle = cycle(&mut executive);
        let mut left_workspace = workspace();
        left_workspace.hand = HandSide::Left;
        let rejection = bind_humanoid_spatial_goal(
            &cycle,
            HumanoidSkillRequirementRole::Manipulation,
            &subject(),
            &left_workspace,
            &goal(HandSide::Left, [1.3, 1.8, 1.2]),
            &policy(HandSide::Right),
            &evidence(HandSide::Right),
            &state(),
            10.05,
            config(),
        )
        .unwrap_err();
        assert_eq!(
            rejection.failure,
            Some(HumanoidSpatialGoalFailure::SelectedHandNotQualified)
        );
    }

    #[test]
    fn stale_target_evidence_is_rejected_without_shrinking_workspace() {
        let mut executive = HumanoidPermitSkillExecutive::new();
        let cycle = cycle(&mut executive);
        let rejection = bind_humanoid_spatial_goal(
            &cycle,
            HumanoidSkillRequirementRole::Manipulation,
            &subject(),
            &workspace(),
            &goal(HandSide::Right, [1.3, 1.8, 1.2]),
            &policy(HandSide::Right),
            &evidence(HandSide::Right),
            &state(),
            10.5,
            config(),
        )
        .unwrap_err();
        assert_eq!(rejection.failure, Some(HumanoidSpatialGoalFailure::GoalEvidenceStale));
    }
}
