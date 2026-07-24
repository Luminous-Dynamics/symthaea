// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic degraded-mode annunciation and workload control.
//!
//! This is not a cockpit display implementation. It is the policy layer that
//! turns machine states into stable, prioritized, rate-limited alerts with
//! explicit corrective actions, acknowledgement semantics, and phase
//! inhibition. It prevents contradictory or chattering mode announcements from
//! becoming a new operational hazard.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum AlertSeverity {
    Advisory,
    Caution,
    Warning,
    Critical,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum HumanFactorsFlightPhase {
    Preflight,
    Takeoff,
    Climb,
    Cruise,
    Search,
    Approach,
    Landing,
    Ground,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum DegradedModeKind {
    NavigationLimited,
    YawLimited,
    VerticalLimited,
    SingleAxisControl,
    ImmediateLanding,
    Autorotation,
    BaselineController,
    EvidenceDegraded,
    MaintenanceRequired,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AlertDefinition {
    pub alert_id: String,
    pub mode: DegradedModeKind,
    pub severity: AlertSeverity,
    pub message: String,
    pub corrective_action: String,
    pub inhibited_phases: Vec<HumanFactorsFlightPhase>,
    pub latched: bool,
    pub acknowledgement_required: bool,
    pub minimum_repeat_interval_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DegradedModeObservation {
    pub timestamp_ms: u64,
    pub flight_phase: HumanFactorsFlightPhase,
    pub active_modes: Vec<DegradedModeKind>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActiveAlert {
    pub alert_id: String,
    pub mode: DegradedModeKind,
    pub severity: AlertSeverity,
    pub message: String,
    pub corrective_action: String,
    pub first_active_ms: u64,
    pub last_announced_ms: u64,
    pub acknowledged: bool,
    pub latched: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HumanFactorsIssue {
    ConflictingPrimaryModes(Vec<DegradedModeKind>),
    AlertCapacityExceeded { active: usize, maximum: usize },
    MissingDefinition(DegradedModeKind),
    DuplicateDefinition(DegradedModeKind),
    InvalidDefinition(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HumanFactorsStatus {
    Nominal,
    Degraded,
    Unsafe,
    Incomplete,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AnnunciationFrame {
    pub timestamp_ms: u64,
    pub status: HumanFactorsStatus,
    pub primary_alert: Option<ActiveAlert>,
    pub displayed_alerts: Vec<ActiveAlert>,
    pub newly_announced: Vec<String>,
    pub suppressed_alerts: Vec<String>,
    pub issues: Vec<HumanFactorsIssue>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HumanFactorsConfig {
    pub schema_version: String,
    pub manager_id: String,
    pub maximum_displayed_alerts: usize,
    pub definitions: Vec<AlertDefinition>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HumanFactorsError {
    InvalidConfiguration,
    UnknownAlert(String),
}

#[derive(Debug, Clone)]
pub struct DegradedModeAnnunciator {
    config: HumanFactorsConfig,
    definitions: BTreeMap<DegradedModeKind, AlertDefinition>,
    active: BTreeMap<DegradedModeKind, ActiveAlert>,
}

impl DegradedModeAnnunciator {
    pub fn new(config: HumanFactorsConfig) -> Result<Self, HumanFactorsError> {
        if config.schema_version.trim().is_empty()
            || config.manager_id.trim().is_empty()
            || config.maximum_displayed_alerts == 0
            || config.definitions.is_empty()
        {
            return Err(HumanFactorsError::InvalidConfiguration);
        }
        let mut definitions = BTreeMap::new();
        for definition in &config.definitions {
            if definition.alert_id.trim().is_empty()
                || definition.message.trim().is_empty()
                || definition.corrective_action.trim().is_empty()
                || definition.minimum_repeat_interval_ms == 0
                || definitions
                    .insert(definition.mode, definition.clone())
                    .is_some()
            {
                return Err(HumanFactorsError::InvalidConfiguration);
            }
        }
        Ok(Self {
            config,
            definitions,
            active: BTreeMap::new(),
        })
    }

    pub fn acknowledge(&mut self, alert_id: &str) -> Result<(), HumanFactorsError> {
        let Some(alert) = self
            .active
            .values_mut()
            .find(|alert| alert.alert_id == alert_id)
        else {
            return Err(HumanFactorsError::UnknownAlert(alert_id.to_string()));
        };
        alert.acknowledged = true;
        Ok(())
    }

    pub fn clear_latched(&mut self, mode: DegradedModeKind) {
        if self
            .active
            .get(&mode)
            .is_some_and(|alert| alert.latched && alert.acknowledged)
        {
            self.active.remove(&mode);
        }
    }

    pub fn update(&mut self, observation: &DegradedModeObservation) -> AnnunciationFrame {
        let requested: BTreeSet<_> = observation.active_modes.iter().copied().collect();
        let mut issues = Vec::new();
        let mut suppressed = Vec::new();
        let mut newly_announced = Vec::new();

        for mode in requested.iter().copied() {
            let Some(definition) = self.definitions.get(&mode) else {
                issues.push(HumanFactorsIssue::MissingDefinition(mode));
                continue;
            };
            if definition
                .inhibited_phases
                .contains(&observation.flight_phase)
            {
                suppressed.push(definition.alert_id.clone());
                continue;
            }
            let alert = self.active.entry(mode).or_insert_with(|| ActiveAlert {
                alert_id: definition.alert_id.clone(),
                mode,
                severity: definition.severity,
                message: definition.message.clone(),
                corrective_action: definition.corrective_action.clone(),
                first_active_ms: observation.timestamp_ms,
                last_announced_ms: 0,
                acknowledged: false,
                latched: definition.latched,
            });
            let due = alert.last_announced_ms == 0
                || observation
                    .timestamp_ms
                    .saturating_sub(alert.last_announced_ms)
                    >= definition.minimum_repeat_interval_ms;
            if due && (!definition.acknowledgement_required || !alert.acknowledged) {
                alert.last_announced_ms = observation.timestamp_ms;
                newly_announced.push(alert.alert_id.clone());
            }
        }

        self.active
            .retain(|mode, alert| requested.contains(mode) || alert.latched);

        let primary_modes: Vec<_> = requested
            .iter()
            .copied()
            .filter(|mode| is_primary_mode(*mode))
            .collect();
        if primary_modes.len() > 1 {
            issues.push(HumanFactorsIssue::ConflictingPrimaryModes(primary_modes));
        }

        let mut displayed: Vec<_> = self.active.values().cloned().collect();
        displayed.sort_by(|a, b| {
            b.severity
                .cmp(&a.severity)
                .then_with(|| a.first_active_ms.cmp(&b.first_active_ms))
                .then_with(|| a.alert_id.cmp(&b.alert_id))
        });
        if displayed.len() > self.config.maximum_displayed_alerts {
            issues.push(HumanFactorsIssue::AlertCapacityExceeded {
                active: displayed.len(),
                maximum: self.config.maximum_displayed_alerts,
            });
            for alert in displayed.drain(self.config.maximum_displayed_alerts..) {
                suppressed.push(alert.alert_id);
            }
        }
        let primary_alert = displayed.first().cloned();
        let status = if issues.iter().any(|issue| {
            matches!(
                issue,
                HumanFactorsIssue::ConflictingPrimaryModes(_)
                    | HumanFactorsIssue::AlertCapacityExceeded { .. }
            )
        }) {
            HumanFactorsStatus::Unsafe
        } else if issues
            .iter()
            .any(|issue| matches!(issue, HumanFactorsIssue::MissingDefinition(_)))
        {
            HumanFactorsStatus::Incomplete
        } else if displayed.is_empty() {
            HumanFactorsStatus::Nominal
        } else {
            HumanFactorsStatus::Degraded
        };

        AnnunciationFrame {
            timestamp_ms: observation.timestamp_ms,
            status,
            primary_alert,
            displayed_alerts: displayed,
            newly_announced,
            suppressed_alerts: suppressed,
            issues,
        }
    }
}

fn is_primary_mode(mode: DegradedModeKind) -> bool {
    matches!(
        mode,
        DegradedModeKind::ImmediateLanding
            | DegradedModeKind::Autorotation
            | DegradedModeKind::SingleAxisControl
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn definition(mode: DegradedModeKind, severity: AlertSeverity) -> AlertDefinition {
        AlertDefinition {
            alert_id: format!("{mode:?}"),
            mode,
            severity,
            message: format!("{mode:?} active"),
            corrective_action: "Follow degraded procedure".into(),
            inhibited_phases: Vec::new(),
            latched: false,
            acknowledgement_required: true,
            minimum_repeat_interval_ms: 1_000,
        }
    }

    fn manager() -> DegradedModeAnnunciator {
        DegradedModeAnnunciator::new(HumanFactorsConfig {
            schema_version: "1".into(),
            manager_id: "crew-alerting".into(),
            maximum_displayed_alerts: 3,
            definitions: vec![
                definition(DegradedModeKind::NavigationLimited, AlertSeverity::Caution),
                definition(DegradedModeKind::ImmediateLanding, AlertSeverity::Critical),
                definition(DegradedModeKind::Autorotation, AlertSeverity::Critical),
            ],
        })
        .unwrap()
    }

    #[test]
    fn critical_alert_becomes_primary() {
        let mut manager = manager();
        let frame = manager.update(&DegradedModeObservation {
            timestamp_ms: 100,
            flight_phase: HumanFactorsFlightPhase::Cruise,
            active_modes: vec![
                DegradedModeKind::NavigationLimited,
                DegradedModeKind::ImmediateLanding,
            ],
        });
        assert_eq!(
            frame.primary_alert.unwrap().mode,
            DegradedModeKind::ImmediateLanding
        );
    }

    #[test]
    fn acknowledgement_stops_repeat_chatter() {
        let mut manager = manager();
        let observation = DegradedModeObservation {
            timestamp_ms: 100,
            flight_phase: HumanFactorsFlightPhase::Cruise,
            active_modes: vec![DegradedModeKind::NavigationLimited],
        };
        let first = manager.update(&observation);
        manager.acknowledge(&first.newly_announced[0]).unwrap();
        let second = manager.update(&DegradedModeObservation {
            timestamp_ms: 5_000,
            ..observation
        });
        assert!(second.newly_announced.is_empty());
    }

    #[test]
    fn conflicting_terminal_modes_are_unsafe() {
        let mut manager = manager();
        let frame = manager.update(&DegradedModeObservation {
            timestamp_ms: 100,
            flight_phase: HumanFactorsFlightPhase::Cruise,
            active_modes: vec![
                DegradedModeKind::ImmediateLanding,
                DegradedModeKind::Autorotation,
            ],
        });
        assert_eq!(frame.status, HumanFactorsStatus::Unsafe);
    }
}
