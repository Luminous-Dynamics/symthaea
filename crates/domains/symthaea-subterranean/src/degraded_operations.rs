// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic degraded-operation supervisor for stale operator links,
//! watchdog failures, reboot loops and invalid recovery state.

use crate::embodiment::MotorSafetyLevel;
use crate::mission::SubterraneanMissionIntent;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DegradedMode {
    Normal,
    OperatorLinkLost,
    AutonomousReturn,
    SafeHold,
    RecoveryRequired,
}

impl DegradedMode {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Normal => "normal",
            Self::OperatorLinkLost => "operator_link_lost",
            Self::AutonomousReturn => "autonomous_return",
            Self::SafeHold => "safe_hold",
            Self::RecoveryRequired => "recovery_required",
        }
    }

    pub const fn mission_override(self) -> Option<SubterraneanMissionIntent> {
        match self {
            Self::Normal | Self::OperatorLinkLost => None,
            Self::AutonomousReturn => Some(SubterraneanMissionIntent::ReturnHome),
            Self::SafeHold | Self::RecoveryRequired => {
                Some(SubterraneanMissionIntent::HoldPosition)
            }
        }
    }

    pub const fn safety_floor(self) -> Option<MotorSafetyLevel> {
        match self {
            Self::Normal => None,
            Self::OperatorLinkLost => Some(MotorSafetyLevel::Yellow),
            Self::AutonomousReturn => Some(MotorSafetyLevel::Yellow),
            Self::SafeHold => Some(MotorSafetyLevel::Orange),
            Self::RecoveryRequired => Some(MotorSafetyLevel::Red),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct DegradedObservation {
    pub operator_link_fresh: bool,
    pub control_loop_healthy: bool,
    pub checkpoint_valid: bool,
    pub reboot_count_in_window: u32,
    pub battery_ratio: f64,
    pub return_feasible: bool,
    pub at_surface_or_service_bay: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DegradedPolicy {
    pub operator_link_grace_steps: u32,
    pub watchdog_failure_limit: u32,
    pub reboot_limit: u32,
    pub recovery_dwell_steps: u32,
}

impl Default for DegradedPolicy {
    fn default() -> Self {
        Self {
            operator_link_grace_steps: 400,
            watchdog_failure_limit: 3,
            reboot_limit: 4,
            recovery_dwell_steps: 200,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct DegradedTransition {
    pub previous: DegradedMode,
    pub current: DegradedMode,
    pub changed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DegradedOperationsSupervisor {
    policy: DegradedPolicy,
    mode: DegradedMode,
    operator_link_loss_steps: u32,
    consecutive_watchdog_failures: u32,
    healthy_recovery_steps: u32,
    transitions: u64,
}

impl DegradedOperationsSupervisor {
    pub fn new(policy: DegradedPolicy) -> Self {
        Self {
            policy,
            mode: DegradedMode::Normal,
            operator_link_loss_steps: 0,
            consecutive_watchdog_failures: 0,
            healthy_recovery_steps: 0,
            transitions: 0,
        }
    }

    pub fn mode(&self) -> DegradedMode {
        self.mode
    }

    pub fn validate(&self) -> bool {
        self.policy.operator_link_grace_steps > 0
            && self.policy.watchdog_failure_limit > 0
            && self.policy.reboot_limit > 0
            && self.policy.recovery_dwell_steps > 0
    }

    pub fn transitions(&self) -> u64 {
        self.transitions
    }

    pub fn operator_link_loss_steps(&self) -> u32 {
        self.operator_link_loss_steps
    }

    pub fn update(&mut self, observation: DegradedObservation) -> DegradedTransition {
        let previous = self.mode;
        self.operator_link_loss_steps = if observation.operator_link_fresh {
            0
        } else {
            self.operator_link_loss_steps.saturating_add(1)
        };
        self.consecutive_watchdog_failures = if observation.control_loop_healthy {
            0
        } else {
            self.consecutive_watchdog_failures.saturating_add(1)
        };

        let fatal_recovery_condition = !observation.checkpoint_valid
            || observation.reboot_count_in_window >= self.policy.reboot_limit
            || self.consecutive_watchdog_failures >= self.policy.watchdog_failure_limit;
        self.mode = if fatal_recovery_condition {
            DegradedMode::RecoveryRequired
        } else if self.operator_link_loss_steps == 0 {
            match self.mode {
                DegradedMode::RecoveryRequired => DegradedMode::RecoveryRequired,
                DegradedMode::SafeHold if !observation.at_surface_or_service_bay => {
                    DegradedMode::SafeHold
                }
                _ => DegradedMode::Normal,
            }
        } else if self.operator_link_loss_steps <= self.policy.operator_link_grace_steps {
            DegradedMode::OperatorLinkLost
        } else if observation.return_feasible
            && observation.battery_ratio.is_finite()
            && observation.battery_ratio >= 0.2
        {
            DegradedMode::AutonomousReturn
        } else {
            DegradedMode::SafeHold
        };

        let changed = self.mode != previous;
        if changed {
            self.transitions = self.transitions.saturating_add(1);
        }
        DegradedTransition {
            previous,
            current: self.mode,
            changed,
        }
    }

    /// RecoveryRequired cannot clear from link restoration alone. A caller must
    /// establish a safe location, a healthy dwell and an explicit external
    /// recovery authorization before calling this method.
    pub fn authorize_recovery_clear(
        &mut self,
        observation: DegradedObservation,
        externally_authorized: bool,
    ) -> bool {
        if self.mode != DegradedMode::RecoveryRequired
            || !externally_authorized
            || !observation.at_surface_or_service_bay
            || !observation.operator_link_fresh
            || !observation.control_loop_healthy
            || !observation.checkpoint_valid
            || observation.reboot_count_in_window >= self.policy.reboot_limit
        {
            self.healthy_recovery_steps = 0;
            return false;
        }
        self.healthy_recovery_steps = self.healthy_recovery_steps.saturating_add(1);
        if self.healthy_recovery_steps >= self.policy.recovery_dwell_steps.max(1) {
            self.mode = DegradedMode::Normal;
            self.healthy_recovery_steps = 0;
            self.consecutive_watchdog_failures = 0;
            self.operator_link_loss_steps = 0;
            self.transitions = self.transitions.saturating_add(1);
            true
        } else {
            false
        }
    }

    pub fn reset_runtime(&mut self) {
        self.mode = DegradedMode::Normal;
        self.operator_link_loss_steps = 0;
        self.consecutive_watchdog_failures = 0;
        self.healthy_recovery_steps = 0;
    }
}

impl Default for DegradedOperationsSupervisor {
    fn default() -> Self {
        Self::new(DegradedPolicy::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn healthy() -> DegradedObservation {
        DegradedObservation {
            operator_link_fresh: true,
            control_loop_healthy: true,
            checkpoint_valid: true,
            reboot_count_in_window: 0,
            battery_ratio: 0.8,
            return_feasible: true,
            at_surface_or_service_bay: false,
        }
    }

    #[test]
    fn prolonged_link_loss_selects_autonomous_return_when_feasible() {
        let mut supervisor = DegradedOperationsSupervisor::new(DegradedPolicy {
            operator_link_grace_steps: 2,
            ..Default::default()
        });
        let mut observation = healthy();
        observation.operator_link_fresh = false;
        supervisor.update(observation);
        supervisor.update(observation);
        let transition = supervisor.update(observation);
        assert_eq!(transition.current, DegradedMode::AutonomousReturn);
    }

    #[test]
    fn link_loss_without_return_margin_selects_safe_hold() {
        let mut supervisor = DegradedOperationsSupervisor::new(DegradedPolicy {
            operator_link_grace_steps: 0,
            ..Default::default()
        });
        let mut observation = healthy();
        observation.operator_link_fresh = false;
        observation.return_feasible = false;
        assert_eq!(
            supervisor.update(observation).current,
            DegradedMode::SafeHold
        );
    }

    #[test]
    fn repeated_watchdog_failures_latch_recovery_required() {
        let mut supervisor = DegradedOperationsSupervisor::new(DegradedPolicy {
            watchdog_failure_limit: 2,
            ..Default::default()
        });
        let mut observation = healthy();
        observation.control_loop_healthy = false;
        supervisor.update(observation);
        assert_eq!(
            supervisor.update(observation).current,
            DegradedMode::RecoveryRequired
        );
        assert_eq!(
            supervisor.update(healthy()).current,
            DegradedMode::RecoveryRequired
        );
    }

    #[test]
    fn recovery_required_needs_authorized_healthy_dwell_at_safe_location() {
        let mut supervisor = DegradedOperationsSupervisor::new(DegradedPolicy {
            watchdog_failure_limit: 1,
            recovery_dwell_steps: 2,
            ..Default::default()
        });
        let mut observation = healthy();
        observation.control_loop_healthy = false;
        supervisor.update(observation);
        let mut recovery = healthy();
        recovery.at_surface_or_service_bay = true;
        assert!(!supervisor.authorize_recovery_clear(recovery, true));
        assert!(supervisor.authorize_recovery_clear(recovery, true));
        assert_eq!(supervisor.mode(), DegradedMode::Normal);
    }
}
