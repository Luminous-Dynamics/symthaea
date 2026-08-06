// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Certification check that the maintenance lifecycle escalates correctly
//! from nominal service through to retirement, and that retirement never
//! regains forward-mission authority.
//!
//! `maintenance.rs` (measured health), `degradation_forecast.rs`
//! (predictive horizon), and `maintenance_window.rs` (the actual
//! disposition ladder) are each unit-tested in isolation, but nothing
//! else in this crate exercises the full lifecycle end to end: a real
//! component wearing down, under a real forecast, driving a real planner,
//! all the way from `Continue` to `RetirementReview`. This module is that
//! integration check.

use crate::degradation_forecast::DegradationForecaster;
use crate::maintenance::{ComponentKind, MaintenanceMonitor, MaintenanceResources, NUM_COMPONENTS};
use crate::maintenance_window::{MaintenanceWindowDisposition, MaintenanceWindowPlanner};
use crate::mission::SubterraneanMissionIntent;
use crate::types::{SubterraneanCommand, SubterraneanState, VIBRATION_LEVEL};
use serde::{Deserialize, Serialize};

/// Upper bound on the simulated lifetime. Generous relative to how many
/// steps the reference wear rate actually needs to reach retirement, so
/// this check is robust to small changes in either the wear model or the
/// forecaster's horizon constants rather than depending on exact timing.
const MAX_LIFECYCLE_STEPS: u64 = 12_000;
/// Health below which this check deliberately depletes spare parts and
/// lubricant, forcing the terminal `RetirementReview` branch once the
/// planner also sees `maintenance_due`. Chosen strictly between the
/// `maintenance_due` (0.55) and `mission_abort_required` (0.22) health
/// thresholds so `ReturnForService`/`HoldForService` is genuinely observed
/// first, with resources still available, before retirement is forced.
const RESOURCE_DEPLETION_HEALTH: f64 = 0.35;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LifecycleGateFailure {
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LifecycleAssuranceReport {
    pub reached_continue: bool,
    pub reached_return_or_hold_for_service: bool,
    pub reached_retirement_review: bool,
    pub retirement_holds_position: bool,
    pub failures: Vec<LifecycleGateFailure>,
}

impl LifecycleAssuranceReport {
    pub fn passed(&self) -> bool {
        self.failures.is_empty()
    }
}

pub struct LifecycleAssuranceValidator;

impl LifecycleAssuranceValidator {
    pub fn run(&self) -> LifecycleAssuranceReport {
        let mut monitor = MaintenanceMonitor::new();
        let mut forecaster = DegradationForecaster::new();
        let mut planner = MaintenanceWindowPlanner::default();

        // Heavy, sustained cutter load under high vibration: a genuine
        // wear trajectory (see MaintenanceMonitor::observe), not a
        // fabricated health value.
        let mut command = SubterraneanCommand::zero();
        command.set_cutter_head(1.0);
        let mut state = SubterraneanState::home();
        state.channels[VIBRATION_LEVEL] = 1.0;

        let mut reached_continue = false;
        let mut reached_return_or_hold = false;
        let mut reached_retirement = false;
        let mut retirement_holds_position = false;

        for _ in 0..MAX_LIFECYCLE_STEPS {
            monitor.observe(&command, &state, 1.0);
            let mut health = [0.0; NUM_COMPONENTS];
            for component in ComponentKind::ALL {
                health[component.index()] = monitor.health(component);
            }
            forecaster.observe(health);
            let forecast = forecaster.forecast();
            let assessment = monitor.assessment();

            let resources = if assessment.minimum_health < RESOURCE_DEPLETION_HEALTH {
                MaintenanceResources {
                    spare_parts_ratio: 0.0,
                    lubricant_ratio: 0.0,
                }
            } else {
                MaintenanceResources::full()
            };

            let outcome = planner.assess(forecast, assessment, resources, false, false);

            match outcome.disposition {
                MaintenanceWindowDisposition::Continue => reached_continue = true,
                MaintenanceWindowDisposition::ReturnForService
                | MaintenanceWindowDisposition::HoldForService => {
                    reached_return_or_hold = true;
                }
                MaintenanceWindowDisposition::FinishBoundedWork => {}
                MaintenanceWindowDisposition::RetirementReview => {
                    reached_retirement = true;
                    retirement_holds_position = outcome.disposition.mission_override()
                        == Some(SubterraneanMissionIntent::HoldPosition);
                    break;
                }
            }
        }

        let mut failures = Vec::new();
        if !reached_continue {
            failures.push(LifecycleGateFailure {
                detail: "component never started in the Continue disposition".into(),
            });
        }
        if !reached_return_or_hold {
            failures.push(LifecycleGateFailure {
                detail: "wear never triggered a service disposition before retirement".into(),
            });
        }
        if !reached_retirement {
            failures.push(LifecycleGateFailure {
                detail: format!(
                    "critical wear with depleted resources never escalated to \
                     RetirementReview within {MAX_LIFECYCLE_STEPS} steps"
                ),
            });
        }
        if reached_retirement && !retirement_holds_position {
            failures.push(LifecycleGateFailure {
                detail: "RetirementReview did not hold position -- the terminal lifecycle \
                          state failed to remove forward-mission authority"
                    .into(),
            });
        }

        LifecycleAssuranceReport {
            reached_continue,
            reached_return_or_hold_for_service: reached_return_or_hold,
            reached_retirement_review: reached_retirement,
            retirement_holds_position,
            failures,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn worn_component_escalates_through_the_full_lifecycle_to_retirement() {
        let report = LifecycleAssuranceValidator.run();
        assert!(report.passed(), "{report:#?}");
        assert!(report.reached_continue);
        assert!(report.reached_return_or_hold_for_service);
        assert!(report.reached_retirement_review);
        assert!(report.retirement_holds_position);
    }
}
