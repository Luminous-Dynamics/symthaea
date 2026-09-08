// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Continuous executive for admitted humanoid skill contracts.
//!
//! A one-time capability admission is not sufficient for embodied execution:
//! terrain, payload retention, consent, authority, qualification, and physical
//! capability can change while a skill is active. This module therefore
//! revalidates the current contract every executive tick and deterministically
//! invalidates goal execution when a required invariant stops holding.
//!
//! The executive does not synthesize motor commands and cannot bypass the
//! prepared-command / final safety path. Protective behavior remains owned by
//! the lower execution stack and may preempt any active goal.

use serde::{Deserialize, Serialize};

use crate::capability_envelope::HumanoidCapabilityEnvelope;
use crate::skill_runtime::{
    HumanoidSkillAdmission, HumanoidSkillContract, HumanoidSkillIntent,
    HumanoidSkillRecoveryPolicy, admit_humanoid_skill_contract,
};

/// Runtime evidence that is semantic rather than joint-level.
///
/// Capability envelopes still own the numeric physical limits. These booleans
/// represent lifecycle facts that a skill executive must not infer from a
/// command vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct HumanoidSkillRuntimeEvidence {
    /// Independent current authority for goal-directed execution.
    pub goal_authority_valid: bool,
    /// A non-zero carried/acquired load is currently retained by the relevant
    /// gripper, fixture, or tool. False is valid before a grasp establishes it.
    pub load_retained: bool,
    /// Human proximity evidence required by a human-interaction skill is live.
    pub human_proximity_valid: bool,
    /// Current explicit consent for the intentional human contact in this skill.
    pub human_contact_consent: bool,
    /// Protective execution currently owns the body. This always preempts goals.
    pub protective_preempted: bool,
    /// The skill-specific goal predicate has been satisfied by the planner /
    /// perception layer. Completion may still require additional postconditions.
    pub objective_satisfied: bool,
}

impl HumanoidSkillRuntimeEvidence {
    pub const fn nominal_goal() -> Self {
        Self {
            goal_authority_valid: true,
            load_retained: false,
            human_proximity_valid: true,
            human_contact_consent: false,
            protective_preempted: false,
            objective_satisfied: false,
        }
    }
}

/// Runtime postconditions are intentionally distinct from start preconditions.
/// In particular, `Grasp` establishes load retention; it must not require the
/// load to already be retained before the grasp begins.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HumanoidSkillPostcondition {
    ObjectiveSatisfied,
    LoadRetentionEstablished,
    HumanContactConsentStillValid,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HumanoidSkillExecutiveState {
    Idle,
    Active,
    Recovering(HumanoidSkillRecoveryPolicy),
    Completed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HumanoidSkillLifecycleViolation {
    GoalAuthorityRevoked,
    ProtectivePreemption,
    CapabilityNoLongerAdmitted,
    LoadRetentionRequiredBeforeStart,
    LoadRetentionLost,
    HumanProximityEvidenceMissing,
    HumanContactConsentMissing,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HumanoidSkillTransition {
    None,
    Started,
    EnteredRecovery(HumanoidSkillLifecycleViolation),
    Completed,
    RecoveryAcknowledged,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HumanoidSkillExecutiveReport {
    pub state: HumanoidSkillExecutiveState,
    pub transition: HumanoidSkillTransition,
    pub postconditions: Vec<HumanoidSkillPostcondition>,
    pub unsatisfied_postconditions: Vec<HumanoidSkillPostcondition>,
    pub admission: Option<HumanoidSkillAdmission>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HumanoidSkillStartRejection {
    pub violation: HumanoidSkillLifecycleViolation,
    pub admission: Option<HumanoidSkillAdmission>,
}

/// A small fail-closed executive around one active semantic skill contract.
///
/// The executive never automatically resumes a goal after a violation or
/// protective preemption. Recovery must be explicitly acknowledged, after which
/// a planner can compile/admit a fresh contract against current evidence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HumanoidSkillExecutive {
    state: HumanoidSkillExecutiveState,
    active: Option<HumanoidSkillContract>,
    last_transition: HumanoidSkillTransition,
}

impl Default for HumanoidSkillExecutive {
    fn default() -> Self {
        Self::new()
    }
}

impl HumanoidSkillExecutive {
    pub const fn new() -> Self {
        Self {
            state: HumanoidSkillExecutiveState::Idle,
            active: None,
            last_transition: HumanoidSkillTransition::None,
        }
    }

    pub const fn state(&self) -> HumanoidSkillExecutiveState {
        self.state
    }

    pub fn active_contract(&self) -> Option<&HumanoidSkillContract> {
        self.active.as_ref()
    }

    /// Attempt to start a freshly compiled contract.
    ///
    /// Start is reject-only. The request is never weakened or clamped by the
    /// executive. `Grasp` deliberately does not require pre-existing retention;
    /// `Carry` does.
    pub fn start(
        &mut self,
        contract: HumanoidSkillContract,
        envelopes: &[HumanoidCapabilityEnvelope],
        evidence: HumanoidSkillRuntimeEvidence,
    ) -> Result<HumanoidSkillExecutiveReport, HumanoidSkillStartRejection> {
        if self.state != HumanoidSkillExecutiveState::Idle {
            return Err(HumanoidSkillStartRejection {
                violation: HumanoidSkillLifecycleViolation::CapabilityNoLongerAdmitted,
                admission: None,
            });
        }

        if !evidence.goal_authority_valid {
            return Err(HumanoidSkillStartRejection {
                violation: HumanoidSkillLifecycleViolation::GoalAuthorityRevoked,
                admission: None,
            });
        }
        if evidence.protective_preempted {
            return Err(HumanoidSkillStartRejection {
                violation: HumanoidSkillLifecycleViolation::ProtectivePreemption,
                admission: None,
            });
        }
        if requires_retention_before_start(contract.intent) && !evidence.load_retained {
            return Err(HumanoidSkillStartRejection {
                violation: HumanoidSkillLifecycleViolation::LoadRetentionRequiredBeforeStart,
                admission: None,
            });
        }
        if requires_human_proximity(contract.intent) && !evidence.human_proximity_valid {
            return Err(HumanoidSkillStartRejection {
                violation: HumanoidSkillLifecycleViolation::HumanProximityEvidenceMissing,
                admission: None,
            });
        }
        if requires_human_consent(contract.intent) && !evidence.human_contact_consent {
            return Err(HumanoidSkillStartRejection {
                violation: HumanoidSkillLifecycleViolation::HumanContactConsentMissing,
                admission: None,
            });
        }

        let admission = admit_humanoid_skill_contract(contract.clone(), envelopes);
        if !admission.admitted {
            return Err(HumanoidSkillStartRejection {
                violation: HumanoidSkillLifecycleViolation::CapabilityNoLongerAdmitted,
                admission: Some(admission),
            });
        }

        self.active = Some(contract);
        self.state = HumanoidSkillExecutiveState::Active;
        self.last_transition = HumanoidSkillTransition::Started;
        Ok(self.report(Some(admission), evidence))
    }

    /// Revalidate the active contract against current evidence.
    ///
    /// Any invariant violation invalidates the current goal and enters the
    /// contract's declared recovery policy. There is no automatic goal resume.
    pub fn tick(
        &mut self,
        envelopes: &[HumanoidCapabilityEnvelope],
        evidence: HumanoidSkillRuntimeEvidence,
    ) -> HumanoidSkillExecutiveReport {
        self.last_transition = HumanoidSkillTransition::None;

        if self.state != HumanoidSkillExecutiveState::Active {
            return self.report(None, evidence);
        }

        let Some(contract) = self.active.clone() else {
            self.state = HumanoidSkillExecutiveState::Idle;
            return self.report(None, evidence);
        };

        if !evidence.goal_authority_valid {
            return self.enter_recovery(
                contract.recovery,
                HumanoidSkillLifecycleViolation::GoalAuthorityRevoked,
                None,
                evidence,
            );
        }
        if evidence.protective_preempted {
            return self.enter_recovery(
                contract.recovery,
                HumanoidSkillLifecycleViolation::ProtectivePreemption,
                None,
                evidence,
            );
        }

        let admission = admit_humanoid_skill_contract(contract.clone(), envelopes);
        if !admission.admitted {
            return self.enter_recovery(
                contract.recovery,
                HumanoidSkillLifecycleViolation::CapabilityNoLongerAdmitted,
                Some(admission),
                evidence,
            );
        }

        if requires_retention_while_active(contract.intent) && !evidence.load_retained {
            return self.enter_recovery(
                contract.recovery,
                HumanoidSkillLifecycleViolation::LoadRetentionLost,
                Some(admission),
                evidence,
            );
        }
        if requires_human_proximity(contract.intent) && !evidence.human_proximity_valid {
            return self.enter_recovery(
                contract.recovery,
                HumanoidSkillLifecycleViolation::HumanProximityEvidenceMissing,
                Some(admission),
                evidence,
            );
        }
        if requires_human_consent(contract.intent) && !evidence.human_contact_consent {
            return self.enter_recovery(
                contract.recovery,
                HumanoidSkillLifecycleViolation::HumanContactConsentMissing,
                Some(admission),
                evidence,
            );
        }

        let unsatisfied = unsatisfied_postconditions(contract.intent, evidence);
        if evidence.objective_satisfied && unsatisfied.is_empty() {
            self.state = HumanoidSkillExecutiveState::Completed;
            self.last_transition = HumanoidSkillTransition::Completed;
        }
        self.report(Some(admission), evidence)
    }

    /// Recovery completion never resumes the old goal. It clears the contract
    /// and returns to Idle so a planner must compile/admit a fresh skill.
    pub fn acknowledge_recovery_complete(&mut self) -> HumanoidSkillExecutiveReport {
        if matches!(self.state, HumanoidSkillExecutiveState::Recovering(_)) {
            self.state = HumanoidSkillExecutiveState::Idle;
            self.active = None;
            self.last_transition = HumanoidSkillTransition::RecoveryAcknowledged;
        } else {
            self.last_transition = HumanoidSkillTransition::None;
        }
        self.report(None, HumanoidSkillRuntimeEvidence::nominal_goal())
    }

    /// Clear a completed skill after the caller has consumed the completion
    /// event. Completed contracts cannot be restarted in place.
    pub fn clear_completed(&mut self) {
        if self.state == HumanoidSkillExecutiveState::Completed {
            self.state = HumanoidSkillExecutiveState::Idle;
            self.active = None;
            self.last_transition = HumanoidSkillTransition::None;
        }
    }

    fn enter_recovery(
        &mut self,
        recovery: HumanoidSkillRecoveryPolicy,
        violation: HumanoidSkillLifecycleViolation,
        admission: Option<HumanoidSkillAdmission>,
        evidence: HumanoidSkillRuntimeEvidence,
    ) -> HumanoidSkillExecutiveReport {
        self.state = HumanoidSkillExecutiveState::Recovering(recovery);
        self.last_transition = HumanoidSkillTransition::EnteredRecovery(violation);
        self.report(admission, evidence)
    }

    fn report(
        &self,
        admission: Option<HumanoidSkillAdmission>,
        evidence: HumanoidSkillRuntimeEvidence,
    ) -> HumanoidSkillExecutiveReport {
        let postconditions = self
            .active
            .as_ref()
            .map(|contract| postconditions_for(contract.intent))
            .unwrap_or_default();
        let unsatisfied_postconditions = self
            .active
            .as_ref()
            .map(|contract| unsatisfied_postconditions(contract.intent, evidence))
            .unwrap_or_default();
        HumanoidSkillExecutiveReport {
            state: self.state,
            transition: self.last_transition,
            postconditions,
            unsatisfied_postconditions,
            admission,
        }
    }
}

pub fn postconditions_for(intent: HumanoidSkillIntent) -> Vec<HumanoidSkillPostcondition> {
    let mut conditions = vec![HumanoidSkillPostcondition::ObjectiveSatisfied];
    if grasp_establishes_retention(intent) {
        conditions.push(HumanoidSkillPostcondition::LoadRetentionEstablished);
    }
    if requires_human_consent(intent) {
        conditions.push(HumanoidSkillPostcondition::HumanContactConsentStillValid);
    }
    conditions
}

fn unsatisfied_postconditions(
    intent: HumanoidSkillIntent,
    evidence: HumanoidSkillRuntimeEvidence,
) -> Vec<HumanoidSkillPostcondition> {
    postconditions_for(intent)
        .into_iter()
        .filter(|condition| match condition {
            HumanoidSkillPostcondition::ObjectiveSatisfied => !evidence.objective_satisfied,
            HumanoidSkillPostcondition::LoadRetentionEstablished => !evidence.load_retained,
            HumanoidSkillPostcondition::HumanContactConsentStillValid => {
                !evidence.human_contact_consent
            }
        })
        .collect()
}

const fn grasp_establishes_retention(intent: HumanoidSkillIntent) -> bool {
    matches!(
        intent,
        HumanoidSkillIntent::Grasp {
            resulting_total_payload_kg,
            ..
        } if resulting_total_payload_kg > 0.0
    )
}

const fn requires_retention_before_start(intent: HumanoidSkillIntent) -> bool {
    matches!(intent, HumanoidSkillIntent::Carry { .. })
}

const fn requires_retention_while_active(intent: HumanoidSkillIntent) -> bool {
    matches!(intent, HumanoidSkillIntent::Carry { .. })
}

const fn requires_human_proximity(intent: HumanoidSkillIntent) -> bool {
    matches!(intent, HumanoidSkillIntent::AssistHuman { .. })
}

const fn requires_human_consent(intent: HumanoidSkillIntent) -> bool {
    matches!(intent, HumanoidSkillIntent::AssistHuman { .. })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability_envelope::{
        HumanInteractionEvidence, HumanoidNominalCapabilityProfile,
        derive_humanoid_capability_envelope,
    };
    use crate::execution::HumanoidAuthorityEnvelope;
    use crate::morphology::HumanoidMorphology;
    use crate::qualification::HumanoidQualificationSubject;
    use crate::skill_runtime::{
        HumanoidLocomotionMode, HumanoidSkillQualificationSet,
        compile_humanoid_skill_contract,
    };
    use crate::types::{ActuationMode, HumanoidTask};

    fn subject(task: HumanoidTask) -> HumanoidQualificationSubject {
        HumanoidQualificationSubject::new(
            HumanoidMorphology::Dexterous53,
            task,
            ActuationMode::NormalizedTorque,
            "skill-executive-test-v1",
        )
    }

    fn qualifications() -> HumanoidSkillQualificationSet {
        HumanoidSkillQualificationSet::new(vec![
            subject(HumanoidTask::Walk),
            subject(HumanoidTask::Reach),
            subject(HumanoidTask::Grasp),
        ])
    }

    fn envelope(task: HumanoidTask, consent: bool) -> HumanoidCapabilityEnvelope {
        let subject = subject(task);
        let profile = HumanoidNominalCapabilityProfile::new(
            &subject, 2.0, 0.4, 1.0, 1.0, 0.2, 12.0, 80.0, 20.0, 0.8, true,
        );
        derive_humanoid_capability_envelope(
            &subject,
            &profile,
            HumanoidAuthorityEnvelope::fully_admitted(),
            if consent {
                HumanInteractionEvidence {
                    human_present: true,
                    proximity_valid: true,
                    nearest_human_distance_m: 1.0,
                    contact_consent: true,
                }
            } else {
                HumanInteractionEvidence::no_human_present()
            },
        )
    }

    #[test]
    fn grasp_can_start_without_preexisting_load_retention() {
        let contract = compile_humanoid_skill_contract(
            HumanoidSkillIntent::Grasp {
                end_effector_speed_mps: 0.1,
                object_contact_force_n: 10.0,
                resulting_total_payload_kg: 3.0,
            },
            &qualifications(),
        )
        .unwrap();
        let mut executive = HumanoidSkillExecutive::new();
        let report = executive
            .start(
                contract,
                &[envelope(HumanoidTask::Grasp, false)],
                HumanoidSkillRuntimeEvidence::nominal_goal(),
            )
            .unwrap();
        assert_eq!(report.state, HumanoidSkillExecutiveState::Active);
        assert!(report
            .unsatisfied_postconditions
            .contains(&HumanoidSkillPostcondition::LoadRetentionEstablished));
    }

    #[test]
    fn grasp_cannot_complete_until_retention_is_established() {
        let contract = compile_humanoid_skill_contract(
            HumanoidSkillIntent::Grasp {
                end_effector_speed_mps: 0.1,
                object_contact_force_n: 10.0,
                resulting_total_payload_kg: 3.0,
            },
            &qualifications(),
        )
        .unwrap();
        let envelopes = [envelope(HumanoidTask::Grasp, false)];
        let mut executive = HumanoidSkillExecutive::new();
        executive
            .start(
                contract,
                &envelopes,
                HumanoidSkillRuntimeEvidence::nominal_goal(),
            )
            .unwrap();
        let mut evidence = HumanoidSkillRuntimeEvidence::nominal_goal();
        evidence.objective_satisfied = true;
        let waiting = executive.tick(&envelopes, evidence);
        assert_eq!(waiting.state, HumanoidSkillExecutiveState::Active);
        assert!(waiting
            .unsatisfied_postconditions
            .contains(&HumanoidSkillPostcondition::LoadRetentionEstablished));

        evidence.load_retained = true;
        let completed = executive.tick(&envelopes, evidence);
        assert_eq!(completed.state, HumanoidSkillExecutiveState::Completed);
    }

    #[test]
    fn carry_requires_retention_before_start_and_throughout_execution() {
        let contract = compile_humanoid_skill_contract(
            HumanoidSkillIntent::Carry {
                mode: HumanoidLocomotionMode::Walk,
                horizontal_speed_mps: 0.5,
                turn_rate_rad_s: 0.1,
                manipulation_speed_mps: 0.1,
                retention_force_n: 10.0,
                resulting_total_payload_kg: 3.0,
            },
            &qualifications(),
        )
        .unwrap();
        let envelopes = [
            envelope(HumanoidTask::Grasp, false),
            envelope(HumanoidTask::Walk, false),
        ];
        let mut executive = HumanoidSkillExecutive::new();
        let rejected = executive
            .start(
                contract.clone(),
                &envelopes,
                HumanoidSkillRuntimeEvidence::nominal_goal(),
            )
            .unwrap_err();
        assert_eq!(
            rejected.violation,
            HumanoidSkillLifecycleViolation::LoadRetentionRequiredBeforeStart
        );

        let mut evidence = HumanoidSkillRuntimeEvidence::nominal_goal();
        evidence.load_retained = true;
        executive.start(contract, &envelopes, evidence).unwrap();
        evidence.load_retained = false;
        let report = executive.tick(&envelopes, evidence);
        assert_eq!(
            report.state,
            HumanoidSkillExecutiveState::Recovering(
                HumanoidSkillRecoveryPolicy::SecureLoadThenReplan
            )
        );
        assert_eq!(
            report.transition,
            HumanoidSkillTransition::EnteredRecovery(
                HumanoidSkillLifecycleViolation::LoadRetentionLost
            )
        );
    }

    #[test]
    fn consent_loss_invalidates_assist_human_immediately() {
        let contract = compile_humanoid_skill_contract(
            HumanoidSkillIntent::AssistHuman {
                end_effector_speed_mps: 0.1,
                human_contact_force_n: 5.0,
            },
            &qualifications(),
        )
        .unwrap();
        let envelopes = [envelope(HumanoidTask::Reach, true)];
        let mut evidence = HumanoidSkillRuntimeEvidence::nominal_goal();
        evidence.human_contact_consent = true;
        evidence.human_proximity_valid = true;
        let mut executive = HumanoidSkillExecutive::new();
        executive.start(contract, &envelopes, evidence).unwrap();

        evidence.human_contact_consent = false;
        let report = executive.tick(&envelopes, evidence);
        assert_eq!(
            report.state,
            HumanoidSkillExecutiveState::Recovering(
                HumanoidSkillRecoveryPolicy::WithdrawThenReplan
            )
        );
        assert_eq!(
            report.transition,
            HumanoidSkillTransition::EnteredRecovery(
                HumanoidSkillLifecycleViolation::HumanContactConsentMissing
            )
        );
    }

    #[test]
    fn capability_loss_invalidates_skill_without_clamping_or_resume() {
        let contract = compile_humanoid_skill_contract(
            HumanoidSkillIntent::Locomote {
                mode: HumanoidLocomotionMode::Walk,
                horizontal_speed_mps: 0.5,
                turn_rate_rad_s: 0.1,
            },
            &qualifications(),
        )
        .unwrap();
        let mut envelopes = vec![envelope(HumanoidTask::Walk, false)];
        let mut executive = HumanoidSkillExecutive::new();
        let evidence = HumanoidSkillRuntimeEvidence::nominal_goal();
        executive.start(contract, &envelopes, evidence).unwrap();

        envelopes[0].limits.max_horizontal_speed_mps = 0.2;
        let report = executive.tick(&envelopes, evidence);
        assert!(matches!(
            report.transition,
            HumanoidSkillTransition::EnteredRecovery(
                HumanoidSkillLifecycleViolation::CapabilityNoLongerAdmitted
            )
        ));

        let ack = executive.acknowledge_recovery_complete();
        assert_eq!(ack.state, HumanoidSkillExecutiveState::Idle);
        assert!(executive.active_contract().is_none());
    }

    #[test]
    fn protective_preemption_invalidates_goal_and_requires_fresh_replan() {
        let contract = compile_humanoid_skill_contract(
            HumanoidSkillIntent::Reach {
                end_effector_speed_mps: 0.1,
            },
            &qualifications(),
        )
        .unwrap();
        let envelopes = [envelope(HumanoidTask::Reach, false)];
        let mut executive = HumanoidSkillExecutive::new();
        let mut evidence = HumanoidSkillRuntimeEvidence::nominal_goal();
        executive.start(contract, &envelopes, evidence).unwrap();
        evidence.protective_preempted = true;
        let report = executive.tick(&envelopes, evidence);
        assert!(matches!(
            report.transition,
            HumanoidSkillTransition::EnteredRecovery(
                HumanoidSkillLifecycleViolation::ProtectivePreemption
            )
        ));
    }
}
