// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Liveness recovery layered over resource-conflict assessment.

use crate::arbitration_deadlock::{
    ArbitrationDeadlockAssessment, ArbitrationDeadlockMonitor, DeadlockDisposition,
};
use crate::arbitration_progress::ArbitrationProgressFrame;
use crate::mission_shedding::MissionSheddingPlan;
use crate::objective_budget::ConflictObjective;
use crate::types::SubterraneanCommand;
use serde::{Deserialize, Serialize};

pub const ARBITRATION_RECOVERY_SCHEMA_VERSION: u16 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ArbitrationRecoveryAuthority {
    Nominal,
    ShedDiscretionary,
    ReturnOnly,
    HoldForReview,
}

impl ArbitrationRecoveryAuthority {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Nominal => "nominal",
            Self::ShedDiscretionary => "shed_discretionary",
            Self::ReturnOnly => "return_only",
            Self::HoldForReview => "hold_for_review",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArbitrationRecoveryAssessment {
    pub authority: ArbitrationRecoveryAuthority,
    pub deadlock: ArbitrationDeadlockAssessment,
    pub shedding: MissionSheddingPlan,
    pub recovery_attempts: u32,
    pub reasons: Vec<String>,
}

impl ArbitrationRecoveryAssessment {
    pub fn nominal() -> Self {
        Self {
            authority: ArbitrationRecoveryAuthority::Nominal,
            deadlock: ArbitrationDeadlockAssessment::nominal(),
            shedding: MissionSheddingPlan::derive(&[], 0),
            recovery_attempts: 0,
            reasons: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArbitrationRecoverySupervisor {
    schema_version: u16,
    deadlock: ArbitrationDeadlockMonitor,
    assessment: ArbitrationRecoveryAssessment,
}

impl Default for ArbitrationRecoverySupervisor {
    fn default() -> Self {
        Self {
            schema_version: ARBITRATION_RECOVERY_SCHEMA_VERSION,
            deadlock: ArbitrationDeadlockMonitor::default(),
            assessment: ArbitrationRecoveryAssessment::nominal(),
        }
    }
}

impl ArbitrationRecoverySupervisor {
    pub fn validate(&self) -> bool {
        self.schema_version == ARBITRATION_RECOVERY_SCHEMA_VERSION
            && self.deadlock.validate()
            && self.assessment.shedding.validate()
            && self.assessment.reasons.len() <= 8
    }

    pub fn observe(
        &mut self,
        frame: ArbitrationProgressFrame,
        active: &[ConflictObjective],
        return_feasible: bool,
    ) -> ArbitrationRecoveryAssessment {
        let deadlock = self.deadlock.observe(frame);
        let mut authority = ArbitrationRecoveryAuthority::Nominal;
        let mut recovery_attempts = self.assessment.recovery_attempts;
        let mut reasons = Vec::new();
        let maximum_discretionary = match deadlock.disposition {
            DeadlockDisposition::Nominal => active.len(),
            DeadlockDisposition::Warning => {
                authority = ArbitrationRecoveryAuthority::ShedDiscretionary;
                recovery_attempts = recovery_attempts.saturating_add(1);
                reasons.push("arbitration_stall_sheds_discretionary_work".to_string());
                1
            }
            DeadlockDisposition::Critical => {
                recovery_attempts = recovery_attempts.saturating_add(1);
                if return_feasible {
                    authority = ArbitrationRecoveryAuthority::ReturnOnly;
                    reasons.push("persistent_deadlock_requires_return".to_string());
                } else {
                    authority = ArbitrationRecoveryAuthority::HoldForReview;
                    reasons.push("persistent_deadlock_without_feasible_return".to_string());
                }
                0
            }
            DeadlockDisposition::Invalid => {
                authority = ArbitrationRecoveryAuthority::HoldForReview;
                reasons.push("invalid_arbitration_progress_evidence".to_string());
                0
            }
        };
        let shedding = MissionSheddingPlan::derive(active, maximum_discretionary);
        self.assessment = ArbitrationRecoveryAssessment {
            authority,
            deadlock,
            shedding,
            recovery_attempts,
            reasons,
        };
        self.assessment.clone()
    }

    pub fn constrain_command(&self, mut command: SubterraneanCommand) -> SubterraneanCommand {
        match self.assessment.authority {
            ArbitrationRecoveryAuthority::Nominal => {}
            ArbitrationRecoveryAuthority::ShedDiscretionary => {
                command.set_cutter_head(command.cutter_head().clamp(0.0, 0.15));
                command.set_auger_feed(command.auger_feed().clamp(0.0, 0.15));
            }
            ArbitrationRecoveryAuthority::ReturnOnly => {
                command.set_cutter_head(0.0);
                command.set_auger_feed(0.0);
                command.set_left_track(command.left_track().min(0.0));
                command.set_right_track(command.right_track().min(0.0));
                command.set_ballast_trim(0.0);
            }
            ArbitrationRecoveryAuthority::HoldForReview => {
                command.set_cutter_head(0.0);
                command.set_auger_feed(0.0);
                command.set_left_track(0.0);
                command.set_right_track(0.0);
                command.set_ballast_trim(0.0);
            }
        }
        command.sanitize();
        command
    }

    pub fn assessment(&self) -> &ArbitrationRecoveryAssessment {
        &self.assessment
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arbitration_progress::ARBITRATION_PROGRESS_SCHEMA_VERSION;

    fn frame(step: u64) -> ArbitrationProgressFrame {
        ArbitrationProgressFrame {
            schema_version: ARBITRATION_PROGRESS_SCHEMA_VERSION,
            step,
            battery_ratio: 0.7,
            return_margin: 0.3,
            hazard_severity: 0.0,
            selected: vec![ConflictObjective::MissionWork],
            completed_work_orders: 0,
            restoration_progress: 0.0,
        }
    }

    #[test]
    fn warning_sheds_discretionary_work_before_forcing_return() {
        let mut supervisor = ArbitrationRecoverySupervisor::default();
        for step in 1..=51 {
            let _ = supervisor.observe(
                frame(step),
                &[ConflictObjective::MissionWork, ConflictObjective::Communications],
                true,
            );
        }
        assert_eq!(
            supervisor.assessment().authority,
            ArbitrationRecoveryAuthority::ShedDiscretionary
        );
        assert!(!supervisor.assessment().shedding.shed.is_empty());
    }
}
