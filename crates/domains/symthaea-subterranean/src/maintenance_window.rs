// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Predictive maintenance-window planning.
//!
//! The planner converts measured health and conservative degradation horizons
//! into mission restrictions. It never schedules productive work on its own and
//! cannot restore authority removed by physical safety or component isolation.

use crate::degradation_forecast::{DegradationForecast, ForecastDisposition};
use crate::maintenance::{MaintenanceAssessment, MaintenanceResources};
use crate::mission::SubterraneanMissionIntent;
use serde::{Deserialize, Serialize};

pub const MAINTENANCE_WINDOW_SCHEMA_VERSION: u16 = 1;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MaintenanceWindowDisposition {
    Continue,
    FinishBoundedWork,
    ReturnForService,
    HoldForService,
    RetirementReview,
}

impl MaintenanceWindowDisposition {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Continue => "continue",
            Self::FinishBoundedWork => "finish_bounded_work",
            Self::ReturnForService => "return_for_service",
            Self::HoldForService => "hold_for_service",
            Self::RetirementReview => "retirement_review",
        }
    }

    pub const fn mission_override(self) -> Option<SubterraneanMissionIntent> {
        match self {
            Self::Continue | Self::FinishBoundedWork => None,
            Self::ReturnForService => Some(SubterraneanMissionIntent::ReturnHome),
            Self::HoldForService | Self::RetirementReview => {
                Some(SubterraneanMissionIntent::HoldPosition)
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MaintenanceWindowPolicy {
    pub minimum_spares_for_service: f64,
    pub minimum_lubricant_for_service: f64,
    pub bounded_finish_horizon_steps: u64,
    pub service_return_horizon_steps: u64,
}

impl Default for MaintenanceWindowPolicy {
    fn default() -> Self {
        Self {
            minimum_spares_for_service: 0.12,
            minimum_lubricant_for_service: 0.05,
            bounded_finish_horizon_steps: 2_000,
            service_return_horizon_steps: 10_000,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MaintenanceWindowAssessment {
    pub disposition: MaintenanceWindowDisposition,
    pub service_resources_available: bool,
    pub at_service_location: bool,
    pub active_work: bool,
    pub forecast: DegradationForecast,
}

impl MaintenanceWindowAssessment {
    pub const fn nominal() -> Self {
        Self {
            disposition: MaintenanceWindowDisposition::Continue,
            service_resources_available: true,
            at_service_location: true,
            active_work: false,
            forecast: DegradationForecast::warming_up(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaintenanceWindowPlanner {
    pub schema_version: u16,
    policy: MaintenanceWindowPolicy,
    last: MaintenanceWindowAssessment,
}

impl MaintenanceWindowPlanner {
    pub fn new(policy: MaintenanceWindowPolicy) -> Self {
        Self {
            schema_version: MAINTENANCE_WINDOW_SCHEMA_VERSION,
            policy,
            last: MaintenanceWindowAssessment::nominal(),
        }
    }

    pub fn assess(
        &mut self,
        forecast: DegradationForecast,
        maintenance: MaintenanceAssessment,
        resources: MaintenanceResources,
        at_service_location: bool,
        active_work: bool,
    ) -> MaintenanceWindowAssessment {
        let service_resources_available = resources.spare_parts_ratio
            >= self.policy.minimum_spares_for_service
            && resources.lubricant_ratio >= self.policy.minimum_lubricant_for_service;

        let disposition = if maintenance.mission_abort_required
            || forecast.disposition == ForecastDisposition::AbortRisk
        {
            if !service_resources_available {
                MaintenanceWindowDisposition::RetirementReview
            } else if at_service_location {
                MaintenanceWindowDisposition::HoldForService
            } else {
                MaintenanceWindowDisposition::ReturnForService
            }
        } else if maintenance.maintenance_due
            || forecast.steps_to_service <= self.policy.service_return_horizon_steps
            || forecast.disposition == ForecastDisposition::ServiceSoon
        {
            if !service_resources_available {
                MaintenanceWindowDisposition::RetirementReview
            } else if at_service_location {
                MaintenanceWindowDisposition::HoldForService
            } else if active_work
                && forecast.steps_to_service > self.policy.bounded_finish_horizon_steps
            {
                MaintenanceWindowDisposition::FinishBoundedWork
            } else {
                MaintenanceWindowDisposition::ReturnForService
            }
        } else {
            MaintenanceWindowDisposition::Continue
        };

        self.last = MaintenanceWindowAssessment {
            disposition,
            service_resources_available,
            at_service_location,
            active_work,
            forecast,
        };
        self.last
    }

    pub const fn last(&self) -> MaintenanceWindowAssessment {
        self.last
    }

    pub fn validate(&self) -> bool {
        self.schema_version == MAINTENANCE_WINDOW_SCHEMA_VERSION
            && self.policy.minimum_spares_for_service.is_finite()
            && (0.0..=1.0).contains(&self.policy.minimum_spares_for_service)
            && self.policy.minimum_lubricant_for_service.is_finite()
            && (0.0..=1.0).contains(&self.policy.minimum_lubricant_for_service)
            && self.policy.bounded_finish_horizon_steps <= self.policy.service_return_horizon_steps
            && self.last.forecast.predicted_minimum_health.is_finite()
    }
}

impl Default for MaintenanceWindowPlanner {
    fn default() -> Self {
        Self::new(MaintenanceWindowPolicy::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::degradation_forecast::ForecastDisposition;

    fn forecast(disposition: ForecastDisposition, steps_to_service: u64) -> DegradationForecast {
        DegradationForecast {
            disposition,
            critical_component: None,
            predicted_minimum_health: 0.6,
            steps_to_service,
            steps_to_abort: steps_to_service.saturating_add(10_000),
            confidence: 1.0,
            observations: 20,
        }
    }

    #[test]
    fn service_horizon_preempts_new_work_before_failure() {
        let mut planner = MaintenanceWindowPlanner::default();
        let assessment = planner.assess(
            forecast(ForecastDisposition::ServiceSoon, 100),
            MaintenanceAssessment::nominal(),
            MaintenanceResources::full(),
            false,
            false,
        );
        assert_eq!(
            assessment.disposition,
            MaintenanceWindowDisposition::ReturnForService
        );
    }

    #[test]
    fn missing_service_resources_escalate_to_retirement_review() {
        let mut planner = MaintenanceWindowPlanner::default();
        let assessment = planner.assess(
            forecast(ForecastDisposition::AbortRisk, 0),
            MaintenanceAssessment::nominal(),
            MaintenanceResources {
                spare_parts_ratio: 0.0,
                lubricant_ratio: 0.0,
            },
            true,
            false,
        );
        assert_eq!(
            assessment.disposition,
            MaintenanceWindowDisposition::RetirementReview
        );
    }
}
