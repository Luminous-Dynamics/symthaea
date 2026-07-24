// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Progressive fleet rollout governance with explicit stop and rollback gates.
//!
//! Promotion is based on qualified cohort evidence, not deployment age alone.
//! A missing rollback target, critical incident, configuration drift, stale
//! evidence, or insufficient dwell blocks advancement.

use std::collections::{BTreeMap, BTreeSet};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum FleetRolloutPhase {
    Lab,
    Shadow,
    Canary,
    LimitedFleet,
    BroadFleet,
    Complete,
    Halted,
    RolledBack,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FleetAircraftRolloutStatus {
    Healthy,
    Restricted,
    Failed,
    Incomplete,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FleetRolloutPolicy {
    pub schema_version: String,
    pub policy_id: String,
    pub fleet_id: String,
    pub baseline_deployment_id: String,
    pub candidate_deployment_id: String,
    pub rollback_deployment_id: String,
    pub required_aircraft_by_phase: BTreeMap<FleetRolloutPhase, usize>,
    pub required_dwell_hours_by_phase: BTreeMap<FleetRolloutPhase, f64>,
    pub maximum_restricted_fraction: f64,
    pub maximum_noncritical_incidents: u64,
    pub maximum_twin_divergence_sigma: f64,
    pub maximum_evidence_age_ms: u64,
    pub required_evidence_kinds: Vec<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FleetAircraftRolloutEvidence {
    pub evidence_id: String,
    pub fleet_id: String,
    pub aircraft_id: String,
    pub deployment_id: String,
    pub phase: FleetRolloutPhase,
    pub assessed_at_ms: u64,
    pub dwell_hours: f64,
    pub status: FleetAircraftRolloutStatus,
    pub critical_incident_count: u64,
    pub noncritical_incident_count: u64,
    pub peak_twin_divergence_sigma: f64,
    pub configuration_drift_detected: bool,
    pub evidence_kinds: Vec<String>,
    pub evidence_refs: Vec<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FleetRolloutAction {
    Advance,
    Hold,
    Halt,
    Rollback,
    Complete,
    Incomplete,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum FleetRolloutIssue {
    InvalidEvidenceIdentity(String),
    DuplicateEvidence(String),
    DuplicateAircraft(String),
    FleetMismatch(String),
    DeploymentMismatch(String),
    PhaseMismatch {
        aircraft_id: String,
        observed: FleetRolloutPhase,
        expected: FleetRolloutPhase,
    },
    FutureEvidence(String),
    StaleEvidence {
        aircraft_id: String,
        age_ms: u64,
        maximum_ms: u64,
    },
    InvalidDwell(String),
    MissingEvidenceReference(String),
    MissingEvidenceKind {
        aircraft_id: String,
        kind: String,
    },
    InsufficientAircraft {
        observed: usize,
        required: usize,
    },
    InsufficientDwell {
        aircraft_id: String,
        observed_hours: f64,
        required_hours: f64,
    },
    CriticalIncident(String),
    ExcessiveIncidents {
        observed: u64,
        maximum: u64,
    },
    ExcessiveTwinDivergence {
        aircraft_id: String,
        observed: f64,
        maximum: f64,
    },
    ConfigurationDrift(String),
    AircraftFailed(String),
    AircraftIncomplete(String),
    ExcessiveRestrictedFraction {
        observed: f64,
        maximum: f64,
    },
    RollbackTargetMissing,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FleetRolloutDecision {
    pub schema_version: String,
    pub policy_id: String,
    pub fleet_id: String,
    pub assessed_at_ms: u64,
    pub current_phase: FleetRolloutPhase,
    pub next_phase: Option<FleetRolloutPhase>,
    pub action: FleetRolloutAction,
    pub candidate_deployment_id: String,
    pub rollback_deployment_id: String,
    pub aircraft_count: usize,
    pub restricted_fraction: f64,
    pub issues: Vec<FleetRolloutIssue>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FleetRolloutError {
    InvalidPolicy,
}

#[derive(Debug, Clone)]
pub struct FleetRolloutGate {
    policy: FleetRolloutPolicy,
}

impl FleetRolloutGate {
    pub fn new(policy: FleetRolloutPolicy) -> Result<Self, FleetRolloutError> {
        let evidence_kinds: BTreeSet<_> = policy.required_evidence_kinds.iter().collect();
        let promoted_phases = [
            FleetRolloutPhase::Lab,
            FleetRolloutPhase::Shadow,
            FleetRolloutPhase::Canary,
            FleetRolloutPhase::LimitedFleet,
            FleetRolloutPhase::BroadFleet,
        ];
        if policy.schema_version.trim().is_empty()
            || policy.policy_id.trim().is_empty()
            || policy.fleet_id.trim().is_empty()
            || policy.baseline_deployment_id.trim().is_empty()
            || policy.candidate_deployment_id.trim().is_empty()
            || policy.rollback_deployment_id.trim().is_empty()
            || policy.candidate_deployment_id == policy.baseline_deployment_id
            || policy.rollback_deployment_id != policy.baseline_deployment_id
            || policy.required_evidence_kinds.is_empty()
            || evidence_kinds.len() != policy.required_evidence_kinds.len()
            || policy
                .required_evidence_kinds
                .iter()
                .any(|kind| kind.trim().is_empty())
            || !policy.maximum_restricted_fraction.is_finite()
            || !(0.0..=1.0).contains(&policy.maximum_restricted_fraction)
            || !policy.maximum_twin_divergence_sigma.is_finite()
            || policy.maximum_twin_divergence_sigma <= 0.0
            || policy.maximum_evidence_age_ms == 0
            || promoted_phases.iter().any(|phase| {
                policy
                    .required_aircraft_by_phase
                    .get(phase)
                    .copied()
                    .unwrap_or(0)
                    == 0
                    || policy
                        .required_dwell_hours_by_phase
                        .get(phase)
                        .is_none_or(|hours| !hours.is_finite() || *hours <= 0.0)
            })
        {
            return Err(FleetRolloutError::InvalidPolicy);
        }
        Ok(Self { policy })
    }

    pub fn assess(
        &self,
        current_phase: FleetRolloutPhase,
        evidence: &[FleetAircraftRolloutEvidence],
        available_deployments: &[String],
        now_ms: u64,
    ) -> FleetRolloutDecision {
        let mut issues = Vec::new();
        let deployments: BTreeSet<_> = available_deployments.iter().map(String::as_str).collect();
        if !deployments.contains(self.policy.rollback_deployment_id.as_str()) {
            issues.push(FleetRolloutIssue::RollbackTargetMissing);
        }
        let mut evidence_ids = BTreeSet::new();
        let mut aircraft_ids = BTreeSet::new();
        let mut valid = Vec::new();
        let required_dwell = self
            .policy
            .required_dwell_hours_by_phase
            .get(&current_phase)
            .copied()
            .unwrap_or(0.0);
        for entry in evidence {
            if entry.evidence_id.trim().is_empty() || entry.aircraft_id.trim().is_empty() {
                issues.push(FleetRolloutIssue::InvalidEvidenceIdentity(
                    entry.evidence_id.clone(),
                ));
            }
            if !evidence_ids.insert(entry.evidence_id.clone()) {
                issues.push(FleetRolloutIssue::DuplicateEvidence(
                    entry.evidence_id.clone(),
                ));
            }
            if !aircraft_ids.insert(entry.aircraft_id.clone()) {
                issues.push(FleetRolloutIssue::DuplicateAircraft(
                    entry.aircraft_id.clone(),
                ));
            }
            if entry.fleet_id != self.policy.fleet_id {
                issues.push(FleetRolloutIssue::FleetMismatch(entry.aircraft_id.clone()));
                continue;
            }
            if entry.deployment_id != self.policy.candidate_deployment_id {
                issues.push(FleetRolloutIssue::DeploymentMismatch(
                    entry.aircraft_id.clone(),
                ));
                continue;
            }
            if entry.phase != current_phase {
                issues.push(FleetRolloutIssue::PhaseMismatch {
                    aircraft_id: entry.aircraft_id.clone(),
                    observed: entry.phase,
                    expected: current_phase,
                });
                continue;
            }
            if entry.assessed_at_ms > now_ms {
                issues.push(FleetRolloutIssue::FutureEvidence(entry.aircraft_id.clone()));
                continue;
            }
            let age = now_ms.saturating_sub(entry.assessed_at_ms);
            if age > self.policy.maximum_evidence_age_ms {
                issues.push(FleetRolloutIssue::StaleEvidence {
                    aircraft_id: entry.aircraft_id.clone(),
                    age_ms: age,
                    maximum_ms: self.policy.maximum_evidence_age_ms,
                });
            }
            if !entry.dwell_hours.is_finite() || entry.dwell_hours < 0.0 {
                issues.push(FleetRolloutIssue::InvalidDwell(entry.aircraft_id.clone()));
            } else if entry.dwell_hours < required_dwell {
                issues.push(FleetRolloutIssue::InsufficientDwell {
                    aircraft_id: entry.aircraft_id.clone(),
                    observed_hours: entry.dwell_hours,
                    required_hours: required_dwell,
                });
            }
            if entry.evidence_refs.is_empty()
                || entry
                    .evidence_refs
                    .iter()
                    .any(|reference| reference.trim().is_empty())
            {
                issues.push(FleetRolloutIssue::MissingEvidenceReference(
                    entry.aircraft_id.clone(),
                ));
            }
            let kinds: BTreeSet<_> = entry.evidence_kinds.iter().map(String::as_str).collect();
            for required in &self.policy.required_evidence_kinds {
                if !kinds.contains(required.as_str()) {
                    issues.push(FleetRolloutIssue::MissingEvidenceKind {
                        aircraft_id: entry.aircraft_id.clone(),
                        kind: required.clone(),
                    });
                }
            }
            if entry.critical_incident_count > 0 {
                issues.push(FleetRolloutIssue::CriticalIncident(
                    entry.aircraft_id.clone(),
                ));
            }
            if !entry.peak_twin_divergence_sigma.is_finite()
                || entry.peak_twin_divergence_sigma > self.policy.maximum_twin_divergence_sigma
            {
                issues.push(FleetRolloutIssue::ExcessiveTwinDivergence {
                    aircraft_id: entry.aircraft_id.clone(),
                    observed: entry.peak_twin_divergence_sigma,
                    maximum: self.policy.maximum_twin_divergence_sigma,
                });
            }
            if entry.configuration_drift_detected {
                issues.push(FleetRolloutIssue::ConfigurationDrift(
                    entry.aircraft_id.clone(),
                ));
            }
            match entry.status {
                FleetAircraftRolloutStatus::Healthy | FleetAircraftRolloutStatus::Restricted => {}
                FleetAircraftRolloutStatus::Failed => {
                    issues.push(FleetRolloutIssue::AircraftFailed(entry.aircraft_id.clone()));
                }
                FleetAircraftRolloutStatus::Incomplete => {
                    issues.push(FleetRolloutIssue::AircraftIncomplete(
                        entry.aircraft_id.clone(),
                    ));
                }
            }
            valid.push(entry);
        }

        let required_aircraft = self
            .policy
            .required_aircraft_by_phase
            .get(&current_phase)
            .copied()
            .unwrap_or(0);
        if valid.len() < required_aircraft {
            issues.push(FleetRolloutIssue::InsufficientAircraft {
                observed: valid.len(),
                required: required_aircraft,
            });
        }
        let noncritical_incidents = valid
            .iter()
            .map(|entry| entry.noncritical_incident_count)
            .sum::<u64>();
        if noncritical_incidents > self.policy.maximum_noncritical_incidents {
            issues.push(FleetRolloutIssue::ExcessiveIncidents {
                observed: noncritical_incidents,
                maximum: self.policy.maximum_noncritical_incidents,
            });
        }
        let restricted_count = valid
            .iter()
            .filter(|entry| entry.status == FleetAircraftRolloutStatus::Restricted)
            .count();
        let restricted_fraction = if valid.is_empty() {
            0.0
        } else {
            restricted_count as f64 / valid.len() as f64
        };
        if restricted_fraction > self.policy.maximum_restricted_fraction {
            issues.push(FleetRolloutIssue::ExcessiveRestrictedFraction {
                observed: restricted_fraction,
                maximum: self.policy.maximum_restricted_fraction,
            });
        }

        let rollback = issues.iter().any(|issue| {
            matches!(
                issue,
                FleetRolloutIssue::CriticalIncident(_)
                    | FleetRolloutIssue::ConfigurationDrift(_)
                    | FleetRolloutIssue::AircraftFailed(_)
                    | FleetRolloutIssue::ExcessiveTwinDivergence { .. }
            )
        }) && !issues
            .iter()
            .any(|issue| matches!(issue, FleetRolloutIssue::RollbackTargetMissing));
        let halt = issues.iter().any(|issue| {
            matches!(
                issue,
                FleetRolloutIssue::CriticalIncident(_)
                    | FleetRolloutIssue::ConfigurationDrift(_)
                    | FleetRolloutIssue::AircraftFailed(_)
                    | FleetRolloutIssue::ExcessiveTwinDivergence { .. }
                    | FleetRolloutIssue::RollbackTargetMissing
            )
        });
        let incomplete = issues.iter().any(|issue| {
            matches!(
                issue,
                FleetRolloutIssue::InvalidEvidenceIdentity(_)
                    | FleetRolloutIssue::DuplicateEvidence(_)
                    | FleetRolloutIssue::DuplicateAircraft(_)
                    | FleetRolloutIssue::FleetMismatch(_)
                    | FleetRolloutIssue::DeploymentMismatch(_)
                    | FleetRolloutIssue::PhaseMismatch { .. }
                    | FleetRolloutIssue::FutureEvidence(_)
                    | FleetRolloutIssue::StaleEvidence { .. }
                    | FleetRolloutIssue::InvalidDwell(_)
                    | FleetRolloutIssue::MissingEvidenceReference(_)
                    | FleetRolloutIssue::MissingEvidenceKind { .. }
                    | FleetRolloutIssue::AircraftIncomplete(_)
            )
        });
        let hold = issues.iter().any(|issue| {
            matches!(
                issue,
                FleetRolloutIssue::InsufficientAircraft { .. }
                    | FleetRolloutIssue::InsufficientDwell { .. }
                    | FleetRolloutIssue::ExcessiveIncidents { .. }
                    | FleetRolloutIssue::ExcessiveRestrictedFraction { .. }
            )
        });

        let next_phase = next_phase(current_phase);
        let action = if rollback {
            FleetRolloutAction::Rollback
        } else if halt {
            FleetRolloutAction::Halt
        } else if incomplete {
            FleetRolloutAction::Incomplete
        } else if hold {
            FleetRolloutAction::Hold
        } else {
            match current_phase {
                FleetRolloutPhase::Complete => FleetRolloutAction::Complete,
                FleetRolloutPhase::Halted => FleetRolloutAction::Halt,
                FleetRolloutPhase::RolledBack => FleetRolloutAction::Rollback,
                FleetRolloutPhase::BroadFleet => FleetRolloutAction::Complete,
                _ => FleetRolloutAction::Advance,
            }
        };

        FleetRolloutDecision {
            schema_version: self.policy.schema_version.clone(),
            policy_id: self.policy.policy_id.clone(),
            fleet_id: self.policy.fleet_id.clone(),
            assessed_at_ms: now_ms,
            current_phase,
            next_phase,
            action,
            candidate_deployment_id: self.policy.candidate_deployment_id.clone(),
            rollback_deployment_id: self.policy.rollback_deployment_id.clone(),
            aircraft_count: valid.len(),
            restricted_fraction,
            issues,
        }
    }
}

fn next_phase(phase: FleetRolloutPhase) -> Option<FleetRolloutPhase> {
    match phase {
        FleetRolloutPhase::Lab => Some(FleetRolloutPhase::Shadow),
        FleetRolloutPhase::Shadow => Some(FleetRolloutPhase::Canary),
        FleetRolloutPhase::Canary => Some(FleetRolloutPhase::LimitedFleet),
        FleetRolloutPhase::LimitedFleet => Some(FleetRolloutPhase::BroadFleet),
        FleetRolloutPhase::BroadFleet => Some(FleetRolloutPhase::Complete),
        FleetRolloutPhase::Complete | FleetRolloutPhase::Halted | FleetRolloutPhase::RolledBack => {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gate() -> FleetRolloutGate {
        FleetRolloutGate::new(FleetRolloutPolicy {
            schema_version: "1".into(),
            policy_id: "rollout".into(),
            fleet_id: "fleet-1".into(),
            baseline_deployment_id: "deployment-a".into(),
            candidate_deployment_id: "deployment-b".into(),
            rollback_deployment_id: "deployment-a".into(),
            required_aircraft_by_phase: BTreeMap::from([
                (FleetRolloutPhase::Lab, 1),
                (FleetRolloutPhase::Shadow, 1),
                (FleetRolloutPhase::Canary, 2),
                (FleetRolloutPhase::LimitedFleet, 3),
                (FleetRolloutPhase::BroadFleet, 4),
            ]),
            required_dwell_hours_by_phase: BTreeMap::from([
                (FleetRolloutPhase::Lab, 0.5),
                (FleetRolloutPhase::Shadow, 1.0),
                (FleetRolloutPhase::Canary, 5.0),
                (FleetRolloutPhase::LimitedFleet, 20.0),
                (FleetRolloutPhase::BroadFleet, 50.0),
            ]),
            maximum_restricted_fraction: 0.25,
            maximum_noncritical_incidents: 2,
            maximum_twin_divergence_sigma: 3.0,
            maximum_evidence_age_ms: 1_000,
            required_evidence_kinds: vec!["readiness".into(), "envelope".into()],
        })
        .unwrap()
    }

    fn evidence(id: &str, phase: FleetRolloutPhase) -> FleetAircraftRolloutEvidence {
        FleetAircraftRolloutEvidence {
            evidence_id: format!("evidence-{id}"),
            fleet_id: "fleet-1".into(),
            aircraft_id: id.into(),
            deployment_id: "deployment-b".into(),
            phase,
            assessed_at_ms: 1_000,
            dwell_hours: 10.0,
            status: FleetAircraftRolloutStatus::Healthy,
            critical_incident_count: 0,
            noncritical_incident_count: 0,
            peak_twin_divergence_sigma: 1.0,
            configuration_drift_detected: false,
            evidence_kinds: vec!["readiness".into(), "envelope".into()],
            evidence_refs: vec![format!("ref-{id}")],
        }
    }

    #[test]
    fn healthy_canary_advances() {
        let decision = gate().assess(
            FleetRolloutPhase::Canary,
            &[
                evidence("a", FleetRolloutPhase::Canary),
                evidence("b", FleetRolloutPhase::Canary),
            ],
            &["deployment-a".into(), "deployment-b".into()],
            1_000,
        );
        assert_eq!(decision.action, FleetRolloutAction::Advance);
        assert_eq!(decision.next_phase, Some(FleetRolloutPhase::LimitedFleet));
    }

    #[test]
    fn critical_incident_rolls_back() {
        let mut a = evidence("a", FleetRolloutPhase::Canary);
        a.critical_incident_count = 1;
        let decision = gate().assess(
            FleetRolloutPhase::Canary,
            &[a, evidence("b", FleetRolloutPhase::Canary)],
            &["deployment-a".into(), "deployment-b".into()],
            1_000,
        );
        assert_eq!(decision.action, FleetRolloutAction::Rollback);
    }

    #[test]
    fn missing_rollback_target_halts() {
        let decision = gate().assess(
            FleetRolloutPhase::Canary,
            &[
                evidence("a", FleetRolloutPhase::Canary),
                evidence("b", FleetRolloutPhase::Canary),
            ],
            &["deployment-b".into()],
            1_000,
        );
        assert_eq!(decision.action, FleetRolloutAction::Halt);
    }

    #[test]
    fn complete_phase_still_rolls_back_on_critical_incident() {
        let mut a = evidence("a", FleetRolloutPhase::Complete);
        a.critical_incident_count = 1;
        let decision = gate().assess(
            FleetRolloutPhase::Complete,
            &[a],
            &["deployment-a".into(), "deployment-b".into()],
            1_000,
        );
        assert_eq!(decision.action, FleetRolloutAction::Rollback);
    }

    #[test]
    fn insufficient_cohort_holds() {
        let decision = gate().assess(
            FleetRolloutPhase::Canary,
            &[evidence("a", FleetRolloutPhase::Canary)],
            &["deployment-a".into(), "deployment-b".into()],
            1_000,
        );
        assert_eq!(decision.action, FleetRolloutAction::Hold);
    }
}
