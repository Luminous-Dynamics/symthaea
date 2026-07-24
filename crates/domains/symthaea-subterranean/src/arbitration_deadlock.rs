// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Persistent no-progress detection for resource arbitration.

use crate::arbitration_progress::ArbitrationProgressFrame;
use serde::{Deserialize, Serialize};

pub const ARBITRATION_DEADLOCK_SCHEMA_VERSION: u16 = 1;
pub const DEFAULT_WARNING_STEPS: u32 = 50;
pub const DEFAULT_CRITICAL_STEPS: u32 = 200;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeadlockDisposition {
    Nominal,
    Warning,
    Critical,
    Invalid,
}

impl DeadlockDisposition {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Nominal => "nominal",
            Self::Warning => "warning",
            Self::Critical => "critical",
            Self::Invalid => "invalid",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ArbitrationDeadlockAssessment {
    pub disposition: DeadlockDisposition,
    pub consecutive_no_progress_steps: u32,
    pub total_deadlock_events: u64,
    pub reason: Option<String>,
}

impl ArbitrationDeadlockAssessment {
    pub fn nominal() -> Self {
        Self {
            disposition: DeadlockDisposition::Nominal,
            consecutive_no_progress_steps: 0,
            total_deadlock_events: 0,
            reason: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArbitrationDeadlockMonitor {
    schema_version: u16,
    warning_steps: u32,
    critical_steps: u32,
    previous: Option<ArbitrationProgressFrame>,
    assessment: ArbitrationDeadlockAssessment,
}

impl Default for ArbitrationDeadlockMonitor {
    fn default() -> Self {
        Self {
            schema_version: ARBITRATION_DEADLOCK_SCHEMA_VERSION,
            warning_steps: DEFAULT_WARNING_STEPS,
            critical_steps: DEFAULT_CRITICAL_STEPS,
            previous: None,
            assessment: ArbitrationDeadlockAssessment::nominal(),
        }
    }
}

impl ArbitrationDeadlockMonitor {
    pub fn validate(&self) -> bool {
        self.schema_version == ARBITRATION_DEADLOCK_SCHEMA_VERSION
            && self.warning_steps > 0
            && self.critical_steps >= self.warning_steps
            && self.previous.as_ref().is_none_or(ArbitrationProgressFrame::validate)
    }

    pub fn observe(&mut self, frame: ArbitrationProgressFrame) -> ArbitrationDeadlockAssessment {
        if !frame.validate()
            || self
                .previous
                .as_ref()
                .is_some_and(|previous| frame.step <= previous.step)
        {
            self.assessment.disposition = DeadlockDisposition::Invalid;
            self.assessment.reason = Some("invalid_or_replayed_progress_frame".to_string());
            return self.assessment.clone();
        }
        let progressed = self
            .previous
            .as_ref()
            .is_none_or(|previous| frame.materially_progressed_from(previous));
        if progressed {
            self.assessment.consecutive_no_progress_steps = 0;
            self.assessment.disposition = DeadlockDisposition::Nominal;
            self.assessment.reason = None;
        } else {
            self.assessment.consecutive_no_progress_steps = self
                .assessment
                .consecutive_no_progress_steps
                .saturating_add(1);
            if self.assessment.consecutive_no_progress_steps >= self.critical_steps {
                if self.assessment.disposition != DeadlockDisposition::Critical {
                    self.assessment.total_deadlock_events =
                        self.assessment.total_deadlock_events.saturating_add(1);
                }
                self.assessment.disposition = DeadlockDisposition::Critical;
                self.assessment.reason = Some("persistent_arbitration_deadlock".to_string());
            } else if self.assessment.consecutive_no_progress_steps >= self.warning_steps {
                self.assessment.disposition = DeadlockDisposition::Warning;
                self.assessment.reason = Some("arbitration_progress_stalled".to_string());
            }
        }
        self.previous = Some(frame);
        self.assessment.clone()
    }

    pub fn assessment(&self) -> &ArbitrationDeadlockAssessment {
        &self.assessment
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::objective_budget::ConflictObjective;

    fn frame(step: u64) -> ArbitrationProgressFrame {
        ArbitrationProgressFrame {
            schema_version: crate::arbitration_progress::ARBITRATION_PROGRESS_SCHEMA_VERSION,
            step,
            battery_ratio: 0.8,
            return_margin: 0.3,
            hazard_severity: 0.0,
            selected: vec![ConflictObjective::MissionWork],
            completed_work_orders: 0,
            restoration_progress: 0.0,
        }
    }

    #[test]
    fn repeated_identical_allocations_escalate_to_warning() {
        let mut monitor = ArbitrationDeadlockMonitor::default();
        let _ = monitor.observe(frame(1));
        for step in 2..=51 {
            let assessment = monitor.observe(frame(step));
            if step == 51 {
                assert_eq!(assessment.disposition, DeadlockDisposition::Warning);
            }
        }
    }
}
