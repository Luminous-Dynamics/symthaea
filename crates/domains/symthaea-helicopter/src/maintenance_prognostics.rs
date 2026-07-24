// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Long-horizon maintenance and life-limit evidence.
//!
//! The model intentionally avoids claiming predictive accuracy it does not have.
//! It performs deterministic accounting over declared hours, cycles, starts,
//! exceedances, and equivalent fatigue damage, preserving serial-number and
//! evidence continuity. Missing observations produce Incomplete status.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComponentLifeLimit {
    pub component_type: String,
    pub serial_number: String,
    pub maximum_hours: Option<f64>,
    pub maximum_cycles: Option<u64>,
    pub maximum_starts: Option<u64>,
    pub maximum_equivalent_damage: Option<f64>,
    pub warning_fraction: f64,
    pub mandatory_replacement_fraction: f64,
    pub provenance_evidence_id: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MaintenanceUsageObservation {
    pub observation_id: String,
    pub component_serial_number: String,
    pub start_timestamp_ms: u64,
    pub end_timestamp_ms: u64,
    pub operating_hours: f64,
    pub cycles: u64,
    pub starts: u64,
    pub equivalent_damage: f64,
    pub exceedance_count: u64,
    pub evidence_ids: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MaintenanceDisposition {
    Serviceable,
    InspectionDue,
    ReplacementDue,
    Grounded,
    Incomplete,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MaintenanceIssue {
    InvalidLimit,
    MissingEvidence(String),
    DuplicateObservation(String),
    SerialMismatch { expected: String, observed: String },
    NonMonotonicObservation(String),
    InvalidUsage(String),
    HoursWarning { used: f64, limit: f64 },
    HoursExceeded { used: f64, limit: f64 },
    CyclesWarning { used: u64, limit: u64 },
    CyclesExceeded { used: u64, limit: u64 },
    StartsWarning { used: u64, limit: u64 },
    StartsExceeded { used: u64, limit: u64 },
    DamageWarning { used: f64, limit: f64 },
    DamageExceeded { used: f64, limit: f64 },
    RecordedExceedance(u64),
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ComponentMaintenanceReport {
    pub component_type: String,
    pub serial_number: String,
    pub disposition: MaintenanceDisposition,
    pub accumulated_hours: f64,
    pub accumulated_cycles: u64,
    pub accumulated_starts: u64,
    pub accumulated_equivalent_damage: f64,
    pub exceedance_count: u64,
    pub remaining_hours: Option<f64>,
    pub remaining_cycles: Option<u64>,
    pub remaining_starts: Option<u64>,
    pub remaining_damage: Option<f64>,
    pub issues: Vec<MaintenanceIssue>,
    pub observation_count: usize,
}

impl ComponentMaintenanceReport {
    pub fn canonical_json(&self) -> Result<Vec<u8>, MaintenanceError> {
        serde_json::to_vec(self).map_err(|_| MaintenanceError::SerializationFailed)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FleetMaintenanceReport {
    pub fleet_id: String,
    pub assessed_at_ms: u64,
    pub disposition: MaintenanceDisposition,
    pub components: Vec<ComponentMaintenanceReport>,
    pub serviceable_count: usize,
    pub inspection_due_count: usize,
    pub replacement_due_count: usize,
    pub grounded_count: usize,
    pub incomplete_count: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MaintenanceError {
    InvalidLimit,
    DuplicateSerial(String),
    SerializationFailed,
}

#[derive(Debug, Clone)]
pub struct MaintenanceLifeTracker {
    limits: BTreeMap<String, ComponentLifeLimit>,
}

impl MaintenanceLifeTracker {
    pub fn new(limits: Vec<ComponentLifeLimit>) -> Result<Self, MaintenanceError> {
        if limits.is_empty() {
            return Err(MaintenanceError::InvalidLimit);
        }
        let mut by_serial = BTreeMap::new();
        for limit in limits {
            if !valid_limit(&limit) {
                return Err(MaintenanceError::InvalidLimit);
            }
            let serial = limit.serial_number.clone();
            if by_serial.insert(serial.clone(), limit).is_some() {
                return Err(MaintenanceError::DuplicateSerial(serial));
            }
        }
        Ok(Self { limits: by_serial })
    }

    pub fn assess_component(
        &self,
        serial_number: &str,
        observations: &[MaintenanceUsageObservation],
    ) -> Result<ComponentMaintenanceReport, MaintenanceError> {
        let Some(limit) = self.limits.get(serial_number) else {
            return Err(MaintenanceError::InvalidLimit);
        };
        let mut issues = Vec::new();
        let mut ids = BTreeSet::new();
        let mut previous_end = None;
        let mut hours = 0.0;
        let mut cycles = 0u64;
        let mut starts = 0u64;
        let mut damage = 0.0;
        let mut exceedance_count = 0u64;

        for observation in observations {
            if !ids.insert(observation.observation_id.clone()) {
                issues.push(MaintenanceIssue::DuplicateObservation(
                    observation.observation_id.clone(),
                ));
            }
            if observation.component_serial_number != serial_number {
                issues.push(MaintenanceIssue::SerialMismatch {
                    expected: serial_number.to_string(),
                    observed: observation.component_serial_number.clone(),
                });
                continue;
            }
            if observation.start_timestamp_ms > observation.end_timestamp_ms
                || previous_end.is_some_and(|end| observation.start_timestamp_ms < end)
            {
                issues.push(MaintenanceIssue::NonMonotonicObservation(
                    observation.observation_id.clone(),
                ));
            }
            previous_end = Some(observation.end_timestamp_ms);
            if !observation.operating_hours.is_finite()
                || observation.operating_hours < 0.0
                || !observation.equivalent_damage.is_finite()
                || observation.equivalent_damage < 0.0
            {
                issues.push(MaintenanceIssue::InvalidUsage(
                    observation.observation_id.clone(),
                ));
                continue;
            }
            if observation.evidence_ids.is_empty() {
                issues.push(MaintenanceIssue::MissingEvidence(
                    observation.observation_id.clone(),
                ));
            }
            hours += observation.operating_hours;
            cycles = cycles.saturating_add(observation.cycles);
            starts = starts.saturating_add(observation.starts);
            damage += observation.equivalent_damage;
            exceedance_count = exceedance_count.saturating_add(observation.exceedance_count);
        }

        assess_scalar(
            hours,
            limit.maximum_hours,
            limit.warning_fraction,
            &mut issues,
            |used, max| MaintenanceIssue::HoursWarning { used, limit: max },
            |used, max| MaintenanceIssue::HoursExceeded { used, limit: max },
        );
        assess_integer(
            cycles,
            limit.maximum_cycles,
            limit.warning_fraction,
            &mut issues,
            |used, max| MaintenanceIssue::CyclesWarning { used, limit: max },
            |used, max| MaintenanceIssue::CyclesExceeded { used, limit: max },
        );
        assess_integer(
            starts,
            limit.maximum_starts,
            limit.warning_fraction,
            &mut issues,
            |used, max| MaintenanceIssue::StartsWarning { used, limit: max },
            |used, max| MaintenanceIssue::StartsExceeded { used, limit: max },
        );
        assess_scalar(
            damage,
            limit.maximum_equivalent_damage,
            limit.warning_fraction,
            &mut issues,
            |used, max| MaintenanceIssue::DamageWarning { used, limit: max },
            |used, max| MaintenanceIssue::DamageExceeded { used, limit: max },
        );
        if exceedance_count > 0 {
            issues.push(MaintenanceIssue::RecordedExceedance(exceedance_count));
        }

        let incomplete = issues.iter().any(|issue| {
            matches!(
                issue,
                MaintenanceIssue::MissingEvidence(_)
                    | MaintenanceIssue::DuplicateObservation(_)
                    | MaintenanceIssue::SerialMismatch { .. }
                    | MaintenanceIssue::NonMonotonicObservation(_)
                    | MaintenanceIssue::InvalidUsage(_)
            )
        });
        let exceeded = issues.iter().any(|issue| {
            matches!(
                issue,
                MaintenanceIssue::HoursExceeded { .. }
                    | MaintenanceIssue::CyclesExceeded { .. }
                    | MaintenanceIssue::StartsExceeded { .. }
                    | MaintenanceIssue::DamageExceeded { .. }
            )
        });
        let warning = issues.iter().any(|issue| {
            matches!(
                issue,
                MaintenanceIssue::HoursWarning { .. }
                    | MaintenanceIssue::CyclesWarning { .. }
                    | MaintenanceIssue::StartsWarning { .. }
                    | MaintenanceIssue::DamageWarning { .. }
                    | MaintenanceIssue::RecordedExceedance(_)
            )
        });
        let replacement_due = limit_replacement_due(limit, hours, cycles, starts, damage);
        let disposition = if exceeded {
            MaintenanceDisposition::Grounded
        } else if incomplete {
            MaintenanceDisposition::Incomplete
        } else if replacement_due {
            MaintenanceDisposition::ReplacementDue
        } else if warning {
            MaintenanceDisposition::InspectionDue
        } else {
            MaintenanceDisposition::Serviceable
        };

        Ok(ComponentMaintenanceReport {
            component_type: limit.component_type.clone(),
            serial_number: limit.serial_number.clone(),
            disposition,
            accumulated_hours: hours,
            accumulated_cycles: cycles,
            accumulated_starts: starts,
            accumulated_equivalent_damage: damage,
            exceedance_count,
            remaining_hours: limit
                .maximum_hours
                .map(|maximum| (maximum - hours).max(0.0)),
            remaining_cycles: limit
                .maximum_cycles
                .map(|maximum| maximum.saturating_sub(cycles)),
            remaining_starts: limit
                .maximum_starts
                .map(|maximum| maximum.saturating_sub(starts)),
            remaining_damage: limit
                .maximum_equivalent_damage
                .map(|maximum| (maximum - damage).max(0.0)),
            issues,
            observation_count: observations.len(),
        })
    }

    pub fn assess_fleet(
        &self,
        fleet_id: &str,
        assessed_at_ms: u64,
        observations: &BTreeMap<String, Vec<MaintenanceUsageObservation>>,
    ) -> Result<FleetMaintenanceReport, MaintenanceError> {
        let mut components = Vec::new();
        for serial in self.limits.keys() {
            components.push(self.assess_component(
                serial,
                observations.get(serial).map(Vec::as_slice).unwrap_or(&[]),
            )?);
        }
        components.sort_by(|a, b| a.serial_number.cmp(&b.serial_number));
        let count = |status| {
            components
                .iter()
                .filter(|report| report.disposition == status)
                .count()
        };
        let serviceable_count = count(MaintenanceDisposition::Serviceable);
        let inspection_due_count = count(MaintenanceDisposition::InspectionDue);
        let replacement_due_count = count(MaintenanceDisposition::ReplacementDue);
        let grounded_count = count(MaintenanceDisposition::Grounded);
        let incomplete_count = count(MaintenanceDisposition::Incomplete);
        let disposition = if grounded_count > 0 {
            MaintenanceDisposition::Grounded
        } else if incomplete_count > 0 {
            MaintenanceDisposition::Incomplete
        } else if replacement_due_count > 0 {
            MaintenanceDisposition::ReplacementDue
        } else if inspection_due_count > 0 {
            MaintenanceDisposition::InspectionDue
        } else {
            MaintenanceDisposition::Serviceable
        };
        Ok(FleetMaintenanceReport {
            fleet_id: fleet_id.to_string(),
            assessed_at_ms,
            disposition,
            components,
            serviceable_count,
            inspection_due_count,
            replacement_due_count,
            grounded_count,
            incomplete_count,
        })
    }
}

fn valid_limit(limit: &ComponentLifeLimit) -> bool {
    !limit.component_type.trim().is_empty()
        && !limit.serial_number.trim().is_empty()
        && !limit.provenance_evidence_id.trim().is_empty()
        && limit.warning_fraction.is_finite()
        && limit.mandatory_replacement_fraction.is_finite()
        && (0.0..=1.0).contains(&limit.warning_fraction)
        && (limit.warning_fraction..=1.0).contains(&limit.mandatory_replacement_fraction)
        && [limit.maximum_hours, limit.maximum_equivalent_damage]
            .into_iter()
            .flatten()
            .all(|value| value.is_finite() && value > 0.0)
        && limit.maximum_cycles.is_none_or(|value| value > 0)
        && limit.maximum_starts.is_none_or(|value| value > 0)
        && (limit.maximum_hours.is_some()
            || limit.maximum_cycles.is_some()
            || limit.maximum_starts.is_some()
            || limit.maximum_equivalent_damage.is_some())
}

fn assess_scalar<W, E>(
    used: f64,
    maximum: Option<f64>,
    warning_fraction: f64,
    issues: &mut Vec<MaintenanceIssue>,
    warning: W,
    exceeded: E,
) where
    W: Fn(f64, f64) -> MaintenanceIssue,
    E: Fn(f64, f64) -> MaintenanceIssue,
{
    if let Some(maximum) = maximum {
        if used > maximum {
            issues.push(exceeded(used, maximum));
        } else if used >= maximum * warning_fraction {
            issues.push(warning(used, maximum));
        }
    }
}

fn assess_integer<W, E>(
    used: u64,
    maximum: Option<u64>,
    warning_fraction: f64,
    issues: &mut Vec<MaintenanceIssue>,
    warning: W,
    exceeded: E,
) where
    W: Fn(u64, u64) -> MaintenanceIssue,
    E: Fn(u64, u64) -> MaintenanceIssue,
{
    if let Some(maximum) = maximum {
        if used > maximum {
            issues.push(exceeded(used, maximum));
        } else if used as f64 >= maximum as f64 * warning_fraction {
            issues.push(warning(used, maximum));
        }
    }
}

fn limit_replacement_due(
    limit: &ComponentLifeLimit,
    hours: f64,
    cycles: u64,
    starts: u64,
    damage: f64,
) -> bool {
    limit
        .maximum_hours
        .is_some_and(|maximum| hours >= maximum * limit.mandatory_replacement_fraction)
        || limit.maximum_cycles.is_some_and(|maximum| {
            cycles as f64 >= maximum as f64 * limit.mandatory_replacement_fraction
        })
        || limit.maximum_starts.is_some_and(|maximum| {
            starts as f64 >= maximum as f64 * limit.mandatory_replacement_fraction
        })
        || limit
            .maximum_equivalent_damage
            .is_some_and(|maximum| damage >= maximum * limit.mandatory_replacement_fraction)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tracker() -> MaintenanceLifeTracker {
        MaintenanceLifeTracker::new(vec![ComponentLifeLimit {
            component_type: "main-rotor-hub".into(),
            serial_number: "hub-001".into(),
            maximum_hours: Some(1_000.0),
            maximum_cycles: Some(5_000),
            maximum_starts: None,
            maximum_equivalent_damage: Some(1.0),
            warning_fraction: 0.8,
            mandatory_replacement_fraction: 0.95,
            provenance_evidence_id: "limit-sheet".into(),
        }])
        .unwrap()
    }

    fn observation(hours: f64, damage: f64) -> MaintenanceUsageObservation {
        MaintenanceUsageObservation {
            observation_id: "flight-1".into(),
            component_serial_number: "hub-001".into(),
            start_timestamp_ms: 0,
            end_timestamp_ms: 1,
            operating_hours: hours,
            cycles: 10,
            starts: 1,
            equivalent_damage: damage,
            exceedance_count: 0,
            evidence_ids: vec!["flight-log".into()],
        }
    }

    #[test]
    fn low_usage_is_serviceable() {
        assert_eq!(
            tracker()
                .assess_component("hub-001", &[observation(10.0, 0.01)])
                .unwrap()
                .disposition,
            MaintenanceDisposition::Serviceable
        );
    }

    #[test]
    fn near_limit_requires_replacement() {
        assert_eq!(
            tracker()
                .assess_component("hub-001", &[observation(960.0, 0.96)])
                .unwrap()
                .disposition,
            MaintenanceDisposition::ReplacementDue
        );
    }

    #[test]
    fn exceeded_limit_grounds_component() {
        assert_eq!(
            tracker()
                .assess_component("hub-001", &[observation(1_001.0, 0.5)])
                .unwrap()
                .disposition,
            MaintenanceDisposition::Grounded
        );
    }
}
