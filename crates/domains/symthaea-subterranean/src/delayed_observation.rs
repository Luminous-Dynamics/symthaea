// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Age-aware observation admission.
//!
//! Delayed measurements may remain useful for mapping or diagnosis while being
//! unsafe for immediate control. This module separates those authorities rather
//! than treating every syntactically valid observation as current truth.

use serde::{Deserialize, Serialize};

pub const DELAYED_OBSERVATION_SCHEMA_VERSION: u16 = 1;
pub const MAX_TIMED_OBSERVATIONS: usize = 32;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ObservationPurpose {
    ImmediateControl,
    HazardDetection,
    Mapping,
    Evidence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct TimedObservation {
    pub source: u16,
    pub purpose: ObservationPurpose,
    pub observed_time_ns: u64,
    pub received_time_ns: u64,
    pub uncertainty_ns: u64,
    pub freshness_limit_ns: u64,
    pub sequence: u64,
}

impl TimedObservation {
    pub fn validate(self) -> bool {
        self.freshness_limit_ns > 0
            && self.observed_time_ns <= self.received_time_ns.saturating_add(self.uncertainty_ns)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ObservationAgeDisposition {
    Fresh,
    Degraded,
    HistoricalOnly,
    Rejected,
}

impl ObservationAgeDisposition {
    pub const fn label(self) -> &'static str {
        match self {
            Self::Fresh => "fresh",
            Self::Degraded => "degraded",
            Self::HistoricalOnly => "historical_only",
            Self::Rejected => "rejected",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ObservationTimingIssue {
    None,
    FutureBeyondUncertainty,
    TooOldForPurpose,
    ExcessiveUncertainty,
    Malformed,
}

impl ObservationTimingIssue {
    pub const fn label(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::FutureBeyondUncertainty => "future_beyond_uncertainty",
            Self::TooOldForPurpose => "too_old_for_purpose",
            Self::ExcessiveUncertainty => "excessive_uncertainty",
            Self::Malformed => "malformed",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObservationTimingAssessment {
    pub disposition: ObservationAgeDisposition,
    pub issue: ObservationTimingIssue,
    pub age_ns: u64,
}

impl ObservationTimingAssessment {
    pub const fn fresh() -> Self {
        Self {
            disposition: ObservationAgeDisposition::Fresh,
            issue: ObservationTimingIssue::None,
            age_ns: 0,
        }
    }

    pub const fn immediate_control_allowed(self) -> bool {
        matches!(self.disposition, ObservationAgeDisposition::Fresh)
    }

    pub const fn historical_use_allowed(self) -> bool {
        !matches!(self.disposition, ObservationAgeDisposition::Rejected)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ObservationBatchAssessment {
    pub worst: ObservationAgeDisposition,
    pub fresh: usize,
    pub degraded: usize,
    pub historical_only: usize,
    pub rejected: usize,
    pub maximum_age_ns: u64,
    pub immediate_control_complete: bool,
}

impl ObservationBatchAssessment {
    pub fn nominal() -> Self {
        Self {
            worst: ObservationAgeDisposition::Fresh,
            fresh: 1,
            degraded: 0,
            historical_only: 0,
            rejected: 0,
            maximum_age_ns: 0,
            immediate_control_complete: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DelayedObservationSupervisor {
    schema_version: u16,
    accepted: u64,
    historical_only: u64,
    rejected: u64,
    last: ObservationBatchAssessment,
}

impl Default for DelayedObservationSupervisor {
    fn default() -> Self {
        Self {
            schema_version: DELAYED_OBSERVATION_SCHEMA_VERSION,
            accepted: 0,
            historical_only: 0,
            rejected: 0,
            last: ObservationBatchAssessment::nominal(),
        }
    }
}

impl DelayedObservationSupervisor {
    pub fn validate(&self) -> bool {
        self.schema_version == DELAYED_OBSERVATION_SCHEMA_VERSION
    }

    pub fn assess_one(
        &mut self,
        control_time_ns: u64,
        observation: TimedObservation,
    ) -> ObservationTimingAssessment {
        let assessment = if !observation.validate() {
            ObservationTimingAssessment {
                disposition: ObservationAgeDisposition::Rejected,
                issue: ObservationTimingIssue::Malformed,
                age_ns: 0,
            }
        } else if observation.observed_time_ns
            > control_time_ns.saturating_add(observation.uncertainty_ns)
        {
            ObservationTimingAssessment {
                disposition: ObservationAgeDisposition::Rejected,
                issue: ObservationTimingIssue::FutureBeyondUncertainty,
                age_ns: 0,
            }
        } else {
            let age_ns = control_time_ns.saturating_sub(observation.observed_time_ns);
            if observation.uncertainty_ns > observation.freshness_limit_ns {
                ObservationTimingAssessment {
                    disposition: ObservationAgeDisposition::Rejected,
                    issue: ObservationTimingIssue::ExcessiveUncertainty,
                    age_ns,
                }
            } else if age_ns <= observation.freshness_limit_ns / 2 {
                ObservationTimingAssessment {
                    disposition: ObservationAgeDisposition::Fresh,
                    issue: ObservationTimingIssue::None,
                    age_ns,
                }
            } else if age_ns <= observation.freshness_limit_ns {
                ObservationTimingAssessment {
                    disposition: ObservationAgeDisposition::Degraded,
                    issue: ObservationTimingIssue::TooOldForPurpose,
                    age_ns,
                }
            } else if matches!(
                observation.purpose,
                ObservationPurpose::Mapping | ObservationPurpose::Evidence
            ) && age_ns <= observation.freshness_limit_ns.saturating_mul(4)
            {
                ObservationTimingAssessment {
                    disposition: ObservationAgeDisposition::HistoricalOnly,
                    issue: ObservationTimingIssue::TooOldForPurpose,
                    age_ns,
                }
            } else {
                ObservationTimingAssessment {
                    disposition: ObservationAgeDisposition::Rejected,
                    issue: ObservationTimingIssue::TooOldForPurpose,
                    age_ns,
                }
            }
        };
        match assessment.disposition {
            ObservationAgeDisposition::Fresh | ObservationAgeDisposition::Degraded => {
                self.accepted = self.accepted.saturating_add(1)
            }
            ObservationAgeDisposition::HistoricalOnly => {
                self.historical_only = self.historical_only.saturating_add(1)
            }
            ObservationAgeDisposition::Rejected => {
                self.rejected = self.rejected.saturating_add(1)
            }
        }
        assessment
    }

    pub fn assess_batch(
        &mut self,
        control_time_ns: u64,
        observations: &[TimedObservation],
    ) -> ObservationBatchAssessment {
        if observations.is_empty() || observations.len() > MAX_TIMED_OBSERVATIONS {
            self.rejected = self.rejected.saturating_add(1);
            self.last = ObservationBatchAssessment {
                worst: ObservationAgeDisposition::Rejected,
                fresh: 0,
                degraded: 0,
                historical_only: 0,
                rejected: observations.len().max(1),
                maximum_age_ns: 0,
                immediate_control_complete: false,
            };
            return self.last.clone();
        }
        let mut report = ObservationBatchAssessment {
            worst: ObservationAgeDisposition::Fresh,
            fresh: 0,
            degraded: 0,
            historical_only: 0,
            rejected: 0,
            maximum_age_ns: 0,
            immediate_control_complete: true,
        };
        let mut immediate_control_seen = false;
        for observation in observations.iter().copied() {
            let assessment = self.assess_one(control_time_ns, observation);
            report.worst = report.worst.max(assessment.disposition);
            report.maximum_age_ns = report.maximum_age_ns.max(assessment.age_ns);
            match assessment.disposition {
                ObservationAgeDisposition::Fresh => report.fresh += 1,
                ObservationAgeDisposition::Degraded => report.degraded += 1,
                ObservationAgeDisposition::HistoricalOnly => report.historical_only += 1,
                ObservationAgeDisposition::Rejected => report.rejected += 1,
            }
            if observation.purpose == ObservationPurpose::ImmediateControl {
                immediate_control_seen = true;
                report.immediate_control_complete &= assessment.immediate_control_allowed();
            }
        }
        report.immediate_control_complete &= immediate_control_seen;
        self.last = report.clone();
        report
    }

    pub fn last(&self) -> &ObservationBatchAssessment {
        &self.last
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn observation(purpose: ObservationPurpose, observed_time_ns: u64) -> TimedObservation {
        TimedObservation {
            source: 1,
            purpose,
            observed_time_ns,
            received_time_ns: 1_000,
            uncertainty_ns: 10,
            freshness_limit_ns: 100,
            sequence: 1,
        }
    }

    #[test]
    fn old_mapping_data_remains_historical_but_not_control_authority() {
        let mut supervisor = DelayedObservationSupervisor::default();
        let result = supervisor.assess_one(
            1_250,
            observation(ObservationPurpose::Mapping, 1_000),
        );
        assert_eq!(result.disposition, ObservationAgeDisposition::HistoricalOnly);
        assert!(!result.immediate_control_allowed());
        assert!(result.historical_use_allowed());
    }

    #[test]
    fn stale_hazard_observation_is_rejected_for_live_control() {
        let mut supervisor = DelayedObservationSupervisor::default();
        let result = supervisor.assess_one(
            1_250,
            observation(ObservationPurpose::HazardDetection, 1_000),
        );
        assert_eq!(result.disposition, ObservationAgeDisposition::Rejected);
    }
}
