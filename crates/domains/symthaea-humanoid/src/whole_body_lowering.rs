// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fail-closed lowering coverage for the current humanoid controller.
//!
//! `whole_body_intent` deliberately describes more physical behavior than the
//! legacy task/baseline path can execute today. This boundary makes that gap
//! explicit. It never drops an objective just to obtain a runnable task.
//!
//! The current lowering profile is code truth, not a qualification/safety claim:
//! it can seed upright posture and straight-line Stand/Walk/Run through the
//! existing task controller. Native turn-rate, end-effector, object-contact,
//! human-contact and payload objectives remain unsupported here until a lower
//! controller explicitly implements them.

use crate::skill_runtime::{HumanoidLocomotionMode, HumanoidSkillIntent};
use crate::types::HumanoidTask;
use crate::whole_body_intent::{
    HumanoidWholeBodyMotionIntent, HumanoidWholeBodyObjectiveIR,
};

pub const CURRENT_HUMANOID_LOWERING_PROFILE_ID: &str =
    "symthaea.humanoid.current-task-lowering.v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HumanoidWholeBodyObjectiveKind {
    UprightPosture,
    LocomotionVelocity,
    EndEffectorTarget,
    ObjectContact,
    HumanContact,
    Payload,
}

impl HumanoidWholeBodyObjectiveKind {
    pub const fn of(objective: &HumanoidWholeBodyObjectiveIR) -> Self {
        match objective {
            HumanoidWholeBodyObjectiveIR::UprightPosture => Self::UprightPosture,
            HumanoidWholeBodyObjectiveIR::LocomotionVelocity { .. } => Self::LocomotionVelocity,
            HumanoidWholeBodyObjectiveIR::EndEffectorTarget { .. } => Self::EndEffectorTarget,
            HumanoidWholeBodyObjectiveIR::ObjectContact { .. } => Self::ObjectContact,
            HumanoidWholeBodyObjectiveIR::HumanContact { .. } => Self::HumanContact,
            HumanoidWholeBodyObjectiveIR::Payload { .. } => Self::Payload,
        }
    }
}

/// Exact implementation coverage of the current legacy task/baseline lowering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HumanoidCurrentLoweringCoverage {
    pub profile_id: &'static str,
    pub upright_posture: bool,
    pub straight_locomotion_velocity: bool,
    pub turn_rate: bool,
    pub end_effector_target: bool,
    pub object_contact: bool,
    pub human_contact: bool,
    pub payload: bool,
}

pub const fn current_humanoid_lowering_coverage() -> HumanoidCurrentLoweringCoverage {
    HumanoidCurrentLoweringCoverage {
        profile_id: CURRENT_HUMANOID_LOWERING_PROFILE_ID,
        upright_posture: true,
        straight_locomotion_velocity: true,
        turn_rate: false,
        end_effector_target: false,
        object_contact: false,
        human_contact: false,
        payload: false,
    }
}

/// Non-authoritative seed for the historical task/baseline controller.
///
/// This type deliberately preserves the validation epoch and source intent
/// lineage. It is not itself motor authority and cannot substitute for the
/// fresh permit at the later execution boundary.
#[derive(Debug, Clone, PartialEq)]
pub struct HumanoidCurrentControllerSeed {
    pub validation_epoch: u64,
    pub lowering_profile_id: &'static str,
    pub task: HumanoidTask,
    pub target_speed_mps: f64,
}

#[derive(Debug, Clone, PartialEq)]
pub enum HumanoidWholeBodyLoweringError {
    InvalidIntentLineage,
    UnexpectedSpatialLineage,
    MissingUprightObjective,
    DuplicateUprightObjective,
    DuplicateLocomotionObjective,
    InvalidLocomotionDemand,
    UnsupportedObjective(HumanoidWholeBodyObjectiveKind),
    NonzeroTurnRateUnsupported { requested_rad_s: f64 },
    SourceSkillMismatch,
}

/// Lower only the exact objective subset implemented by the current
/// task/baseline controller.
///
/// The function is intentionally reject-only. Unsupported objectives are never
/// ignored, approximated, or converted into a different semantic action.
pub fn lower_whole_body_intent_for_current_controller(
    intent: &HumanoidWholeBodyMotionIntent,
) -> Result<HumanoidCurrentControllerSeed, HumanoidWholeBodyLoweringError> {
    if intent.validation_epoch == 0
        || intent.backend_profile_id.trim().is_empty()
        || intent.requirement_subject_fingerprints.is_empty()
        || intent
            .requirement_subject_fingerprints
            .iter()
            .any(|fingerprint| *fingerprint == 0)
    {
        return Err(HumanoidWholeBodyLoweringError::InvalidIntentLineage);
    }
    if intent.spatial_goal_id.is_some() {
        return Err(HumanoidWholeBodyLoweringError::UnexpectedSpatialLineage);
    }

    let coverage = current_humanoid_lowering_coverage();
    let mut upright_count = 0usize;
    let mut locomotion: Option<(HumanoidLocomotionMode, f64, f64)> = None;

    for objective in &intent.objectives {
        match objective {
            HumanoidWholeBodyObjectiveIR::UprightPosture => {
                if !coverage.upright_posture {
                    return Err(HumanoidWholeBodyLoweringError::UnsupportedObjective(
                        HumanoidWholeBodyObjectiveKind::UprightPosture,
                    ));
                }
                upright_count += 1;
                if upright_count > 1 {
                    return Err(HumanoidWholeBodyLoweringError::DuplicateUprightObjective);
                }
            }
            HumanoidWholeBodyObjectiveIR::LocomotionVelocity {
                mode,
                horizontal_speed_mps,
                turn_rate_rad_s,
            } => {
                if !coverage.straight_locomotion_velocity {
                    return Err(HumanoidWholeBodyLoweringError::UnsupportedObjective(
                        HumanoidWholeBodyObjectiveKind::LocomotionVelocity,
                    ));
                }
                if locomotion.is_some() {
                    return Err(HumanoidWholeBodyLoweringError::DuplicateLocomotionObjective);
                }
                if !horizontal_speed_mps.is_finite()
                    || *horizontal_speed_mps < 0.0
                    || !turn_rate_rad_s.is_finite()
                {
                    return Err(HumanoidWholeBodyLoweringError::InvalidLocomotionDemand);
                }
                if !coverage.turn_rate && turn_rate_rad_s.abs() > 1e-12 {
                    return Err(HumanoidWholeBodyLoweringError::NonzeroTurnRateUnsupported {
                        requested_rad_s: *turn_rate_rad_s,
                    });
                }
                locomotion = Some((*mode, *horizontal_speed_mps, *turn_rate_rad_s));
            }
            HumanoidWholeBodyObjectiveIR::EndEffectorTarget { .. } => {
                return Err(HumanoidWholeBodyLoweringError::UnsupportedObjective(
                    HumanoidWholeBodyObjectiveKind::EndEffectorTarget,
                ));
            }
            HumanoidWholeBodyObjectiveIR::ObjectContact { .. } => {
                return Err(HumanoidWholeBodyLoweringError::UnsupportedObjective(
                    HumanoidWholeBodyObjectiveKind::ObjectContact,
                ));
            }
            HumanoidWholeBodyObjectiveIR::HumanContact { .. } => {
                return Err(HumanoidWholeBodyLoweringError::UnsupportedObjective(
                    HumanoidWholeBodyObjectiveKind::HumanContact,
                ));
            }
            HumanoidWholeBodyObjectiveIR::Payload { .. } => {
                return Err(HumanoidWholeBodyLoweringError::UnsupportedObjective(
                    HumanoidWholeBodyObjectiveKind::Payload,
                ));
            }
        }
    }

    if upright_count != 1 {
        return Err(HumanoidWholeBodyLoweringError::MissingUprightObjective);
    }

    let (task, target_speed_mps) = match (intent.source_skill, locomotion) {
        (HumanoidSkillIntent::Stand, None) => (HumanoidTask::Stand, 0.0),
        (
            HumanoidSkillIntent::Locomote {
                mode,
                horizontal_speed_mps,
                turn_rate_rad_s,
            },
            Some((lowered_mode, lowered_speed, lowered_turn)),
        ) if mode == lowered_mode
            && horizontal_speed_mps.to_bits() == lowered_speed.to_bits()
            && turn_rate_rad_s.to_bits() == lowered_turn.to_bits() =>
        {
            (
                match mode {
                    HumanoidLocomotionMode::Walk => HumanoidTask::Walk,
                    HumanoidLocomotionMode::Run => HumanoidTask::Run,
                },
                lowered_speed,
            )
        }
        _ => return Err(HumanoidWholeBodyLoweringError::SourceSkillMismatch),
    };

    Ok(HumanoidCurrentControllerSeed {
        validation_epoch: intent.validation_epoch,
        lowering_profile_id: coverage.profile_id,
        task,
        target_speed_mps,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::morphology::{HandSide, HumanoidMorphology};
    use crate::types::ActuationMode;
    use crate::whole_body_intent::{
        HumanoidContactObjectiveMode, HumanoidWholeBodyInvariantIR,
    };

    fn base_intent(
        source_skill: HumanoidSkillIntent,
        objectives: Vec<HumanoidWholeBodyObjectiveIR>,
    ) -> HumanoidWholeBodyMotionIntent {
        HumanoidWholeBodyMotionIntent {
            validation_epoch: 7,
            morphology: HumanoidMorphology::Dexterous53,
            actuation_mode: ActuationMode::NormalizedTorque,
            backend_profile_id: "lowering-test-backend-v1".into(),
            source_skill,
            requirement_subject_fingerprints: vec![41],
            spatial_goal_id: None,
            objectives,
            invariants: vec![HumanoidWholeBodyInvariantIR::CapabilityEnvelopeRemainsAdmitted],
        }
    }

    #[test]
    fn stand_lowers_to_existing_stand_task() {
        let intent = base_intent(
            HumanoidSkillIntent::Stand,
            vec![HumanoidWholeBodyObjectiveIR::UprightPosture],
        );
        let seed = lower_whole_body_intent_for_current_controller(&intent).unwrap();
        assert_eq!(seed.task, HumanoidTask::Stand);
        assert_eq!(seed.target_speed_mps, 0.0);
    }

    #[test]
    fn straight_walk_lowers_with_exact_requested_speed() {
        let intent = base_intent(
            HumanoidSkillIntent::Locomote {
                mode: HumanoidLocomotionMode::Walk,
                horizontal_speed_mps: 0.4,
                turn_rate_rad_s: 0.0,
            },
            vec![
                HumanoidWholeBodyObjectiveIR::UprightPosture,
                HumanoidWholeBodyObjectiveIR::LocomotionVelocity {
                    mode: HumanoidLocomotionMode::Walk,
                    horizontal_speed_mps: 0.4,
                    turn_rate_rad_s: 0.0,
                },
            ],
        );
        let seed = lower_whole_body_intent_for_current_controller(&intent).unwrap();
        assert_eq!(seed.task, HumanoidTask::Walk);
        assert_eq!(seed.target_speed_mps, 0.4);
    }

    #[test]
    fn nonzero_turn_is_not_silently_dropped() {
        let intent = base_intent(
            HumanoidSkillIntent::Locomote {
                mode: HumanoidLocomotionMode::Walk,
                horizontal_speed_mps: 0.4,
                turn_rate_rad_s: 0.2,
            },
            vec![
                HumanoidWholeBodyObjectiveIR::UprightPosture,
                HumanoidWholeBodyObjectiveIR::LocomotionVelocity {
                    mode: HumanoidLocomotionMode::Walk,
                    horizontal_speed_mps: 0.4,
                    turn_rate_rad_s: 0.2,
                },
            ],
        );
        assert_eq!(
            lower_whole_body_intent_for_current_controller(&intent),
            Err(HumanoidWholeBodyLoweringError::NonzeroTurnRateUnsupported {
                requested_rad_s: 0.2,
            })
        );
    }

    #[test]
    fn source_skill_and_objective_must_match_exactly() {
        let intent = base_intent(
            HumanoidSkillIntent::Locomote {
                mode: HumanoidLocomotionMode::Walk,
                horizontal_speed_mps: 0.4,
                turn_rate_rad_s: 0.0,
            },
            vec![
                HumanoidWholeBodyObjectiveIR::UprightPosture,
                HumanoidWholeBodyObjectiveIR::LocomotionVelocity {
                    mode: HumanoidLocomotionMode::Walk,
                    horizontal_speed_mps: 0.5,
                    turn_rate_rad_s: 0.0,
                },
            ],
        );
        assert_eq!(
            lower_whole_body_intent_for_current_controller(&intent),
            Err(HumanoidWholeBodyLoweringError::SourceSkillMismatch)
        );
    }

    #[test]
    fn missing_upright_objective_fails_closed() {
        let intent = base_intent(HumanoidSkillIntent::Stand, vec![]);
        assert_eq!(
            lower_whole_body_intent_for_current_controller(&intent),
            Err(HumanoidWholeBodyLoweringError::MissingUprightObjective)
        );
    }

    #[test]
    fn carry_cannot_be_lowered_by_erasing_manipulation() {
        let intent = base_intent(
            HumanoidSkillIntent::Carry {
                mode: HumanoidLocomotionMode::Walk,
                horizontal_speed_mps: 0.3,
                turn_rate_rad_s: 0.0,
                manipulation_speed_mps: 0.1,
                retention_force_n: 8.0,
                resulting_total_payload_kg: 2.0,
            },
            vec![
                HumanoidWholeBodyObjectiveIR::UprightPosture,
                HumanoidWholeBodyObjectiveIR::LocomotionVelocity {
                    mode: HumanoidLocomotionMode::Walk,
                    horizontal_speed_mps: 0.3,
                    turn_rate_rad_s: 0.0,
                },
                HumanoidWholeBodyObjectiveIR::EndEffectorTarget {
                    hand: HandSide::Right,
                    target_root_m: [0.3, -0.2, 0.2],
                    maximum_speed_mps: 0.1,
                },
                HumanoidWholeBodyObjectiveIR::ObjectContact {
                    hand: HandSide::Right,
                    mode: HumanoidContactObjectiveMode::Maintain,
                    requested_contact_force_n: 8.0,
                    resulting_total_payload_kg: 2.0,
                },
                HumanoidWholeBodyObjectiveIR::Payload {
                    total_payload_kg: 2.0,
                },
            ],
        );
        assert_eq!(
            lower_whole_body_intent_for_current_controller(&intent),
            Err(HumanoidWholeBodyLoweringError::UnsupportedObjective(
                HumanoidWholeBodyObjectiveKind::EndEffectorTarget,
            ))
        );
    }
}
