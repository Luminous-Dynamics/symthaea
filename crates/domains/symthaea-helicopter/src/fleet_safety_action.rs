// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fleet grounding, restriction, inspection, and recall coordination.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum FleetSafetyActionKind {
    Advisory,
    RestrictOperation,
    MandatoryInspection,
    Ground,
    RecallSoftware,
    RecallHardware,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FleetSafetyScope {
    EntireFleet,
    AircraftIds(BTreeSet<String>),
    AirframeModel(String),
    DeploymentDigest(String),
    ComponentPartNumber(String),
    ComponentSerialRange {
        prefix: String,
        first: u64,
        last: u64,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FleetSafetyAction {
    pub action_id: String,
    pub kind: FleetSafetyActionKind,
    pub scope: FleetSafetyScope,
    pub issued_at_ms: u64,
    pub effective_at_ms: u64,
    pub compliance_deadline_ms: u64,
    pub expires_at_ms: Option<u64>,
    pub authority_id: String,
    pub reason_evidence_id: String,
    pub required_remediation_ids: BTreeSet<String>,
    pub authenticity_evidence_id: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FleetAircraftIdentity {
    pub aircraft_id: String,
    pub airframe_model: String,
    pub deployment_digest: String,
    pub component_part_numbers: BTreeSet<String>,
    pub component_serials: BTreeSet<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FleetSafetyComplianceState {
    Unknown,
    Notified,
    Restricted,
    Grounded,
    Remediating,
    Compliant,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FleetSafetyComplianceEvidence {
    pub action_id: String,
    pub aircraft_id: String,
    pub state: FleetSafetyComplianceState,
    pub acknowledged_at_ms: Option<u64>,
    pub remediations_completed: BTreeSet<String>,
    pub evidence_ids: BTreeSet<String>,
    pub assessed_at_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FleetSafetyActionPolicy {
    pub maximum_acknowledgement_delay_ms: u64,
    pub require_authenticity: bool,
    pub require_evidence_for_compliance: bool,
}

impl Default for FleetSafetyActionPolicy {
    fn default() -> Self {
        Self {
            maximum_acknowledgement_delay_ms: 60 * 60 * 1_000,
            require_authenticity: true,
            require_evidence_for_compliance: true,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FleetSafetyActionStatus {
    Pass,
    Fail,
    Incomplete,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FleetSafetyActionIssue {
    InvalidActionWindow,
    MissingAuthority,
    MissingReasonEvidence,
    MissingAuthenticityEvidence,
    DuplicateAircraft {
        aircraft_id: String,
    },
    MissingComplianceEvidence {
        aircraft_id: String,
    },
    LateAcknowledgement {
        aircraft_id: String,
        delay_ms: u64,
    },
    ComplianceDeadlineMissed {
        aircraft_id: String,
    },
    UnsafeStateForAction {
        aircraft_id: String,
        state: FleetSafetyComplianceState,
    },
    MissingRemediation {
        aircraft_id: String,
        remediation_id: String,
    },
    MissingComplianceArtifact {
        aircraft_id: String,
    },
    UnknownAircraftInEvidence {
        aircraft_id: String,
    },
    DuplicateComplianceEvidence {
        aircraft_id: String,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FleetSafetyAircraftAssessment {
    pub aircraft_id: String,
    pub in_scope: bool,
    pub compliant: bool,
    pub required_operational_state: FleetSafetyComplianceState,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FleetSafetyActionReport {
    pub status: FleetSafetyActionStatus,
    pub action_id: String,
    pub in_scope_aircraft: usize,
    pub compliant_aircraft: usize,
    pub aircraft: Vec<FleetSafetyAircraftAssessment>,
    pub issues: Vec<FleetSafetyActionIssue>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FleetSafetyActionError {
    InvalidPolicy,
    EmptyActionId,
}

pub struct FleetSafetyActionCoordinator {
    policy: FleetSafetyActionPolicy,
}

impl FleetSafetyActionCoordinator {
    pub fn new(policy: FleetSafetyActionPolicy) -> Result<Self, FleetSafetyActionError> {
        if policy.maximum_acknowledgement_delay_ms == 0 {
            return Err(FleetSafetyActionError::InvalidPolicy);
        }
        Ok(Self { policy })
    }

    pub fn assess(
        &self,
        action: &FleetSafetyAction,
        fleet: &[FleetAircraftIdentity],
        evidence: &[FleetSafetyComplianceEvidence],
        now_ms: u64,
    ) -> Result<FleetSafetyActionReport, FleetSafetyActionError> {
        if action.action_id.trim().is_empty() {
            return Err(FleetSafetyActionError::EmptyActionId);
        }
        let mut issues = Vec::new();
        if action.effective_at_ms < action.issued_at_ms
            || action.compliance_deadline_ms < action.effective_at_ms
            || action
                .expires_at_ms
                .is_some_and(|expires| expires < action.effective_at_ms)
        {
            issues.push(FleetSafetyActionIssue::InvalidActionWindow);
        }
        if action.authority_id.trim().is_empty() {
            issues.push(FleetSafetyActionIssue::MissingAuthority);
        }
        if action.reason_evidence_id.trim().is_empty() {
            issues.push(FleetSafetyActionIssue::MissingReasonEvidence);
        }
        if self.policy.require_authenticity
            && action
                .authenticity_evidence_id
                .as_deref()
                .unwrap_or("")
                .is_empty()
        {
            issues.push(FleetSafetyActionIssue::MissingAuthenticityEvidence);
        }

        let mut fleet_by_id = BTreeMap::<&str, &FleetAircraftIdentity>::new();
        for aircraft in fleet {
            if fleet_by_id
                .insert(aircraft.aircraft_id.as_str(), aircraft)
                .is_some()
            {
                issues.push(FleetSafetyActionIssue::DuplicateAircraft {
                    aircraft_id: aircraft.aircraft_id.clone(),
                });
            }
        }
        let mut evidence_by_aircraft = BTreeMap::<&str, Vec<&FleetSafetyComplianceEvidence>>::new();
        for item in evidence
            .iter()
            .filter(|item| item.action_id == action.action_id)
        {
            if !fleet_by_id.contains_key(item.aircraft_id.as_str()) {
                issues.push(FleetSafetyActionIssue::UnknownAircraftInEvidence {
                    aircraft_id: item.aircraft_id.clone(),
                });
            }
            evidence_by_aircraft
                .entry(item.aircraft_id.as_str())
                .or_default()
                .push(item);
        }
        for (aircraft_id, matching) in &evidence_by_aircraft {
            if matching.len() > 1 {
                issues.push(FleetSafetyActionIssue::DuplicateComplianceEvidence {
                    aircraft_id: (*aircraft_id).to_string(),
                });
            }
        }

        let required_state = required_state(action.kind);
        let mut aircraft_assessments = Vec::new();
        let mut compliant_aircraft = 0usize;
        for aircraft in fleet {
            let in_scope = scope_matches(&action.scope, aircraft);
            let mut compliant = !in_scope;
            if in_scope {
                let item = evidence_by_aircraft
                    .get(aircraft.aircraft_id.as_str())
                    .and_then(|items| items.first())
                    .copied();
                if let Some(item) = item {
                    compliant = self.assess_aircraft(
                        action,
                        aircraft,
                        item,
                        required_state,
                        now_ms,
                        &mut issues,
                    );
                } else {
                    issues.push(FleetSafetyActionIssue::MissingComplianceEvidence {
                        aircraft_id: aircraft.aircraft_id.clone(),
                    });
                    if now_ms > action.compliance_deadline_ms {
                        issues.push(FleetSafetyActionIssue::ComplianceDeadlineMissed {
                            aircraft_id: aircraft.aircraft_id.clone(),
                        });
                    }
                }
                if compliant {
                    compliant_aircraft += 1;
                }
            }
            aircraft_assessments.push(FleetSafetyAircraftAssessment {
                aircraft_id: aircraft.aircraft_id.clone(),
                in_scope,
                compliant,
                required_operational_state: required_state,
            });
        }
        aircraft_assessments.sort_by(|a, b| a.aircraft_id.cmp(&b.aircraft_id));
        issues.sort_by(|a, b| format!("{a:?}").cmp(&format!("{b:?}")));
        let in_scope_aircraft = aircraft_assessments
            .iter()
            .filter(|item| item.in_scope)
            .count();
        let status = if issues.iter().any(issue_is_failure) {
            FleetSafetyActionStatus::Fail
        } else if issues.is_empty() {
            FleetSafetyActionStatus::Pass
        } else {
            FleetSafetyActionStatus::Incomplete
        };
        Ok(FleetSafetyActionReport {
            status,
            action_id: action.action_id.clone(),
            in_scope_aircraft,
            compliant_aircraft,
            aircraft: aircraft_assessments,
            issues,
        })
    }

    fn assess_aircraft(
        &self,
        action: &FleetSafetyAction,
        aircraft: &FleetAircraftIdentity,
        item: &FleetSafetyComplianceEvidence,
        required_state: FleetSafetyComplianceState,
        now_ms: u64,
        issues: &mut Vec<FleetSafetyActionIssue>,
    ) -> bool {
        let mut compliant = true;
        match item.acknowledged_at_ms {
            Some(acknowledged) => {
                let delay = acknowledged.saturating_sub(action.issued_at_ms);
                if delay > self.policy.maximum_acknowledgement_delay_ms {
                    issues.push(FleetSafetyActionIssue::LateAcknowledgement {
                        aircraft_id: aircraft.aircraft_id.clone(),
                        delay_ms: delay,
                    });
                    compliant = false;
                }
            }
            None => {
                issues.push(FleetSafetyActionIssue::MissingComplianceEvidence {
                    aircraft_id: aircraft.aircraft_id.clone(),
                });
                compliant = false;
            }
        }
        if now_ms > action.compliance_deadline_ms
            && item.state != FleetSafetyComplianceState::Compliant
        {
            issues.push(FleetSafetyActionIssue::ComplianceDeadlineMissed {
                aircraft_id: aircraft.aircraft_id.clone(),
            });
            compliant = false;
        }
        if !state_satisfies(action.kind, item.state) {
            issues.push(FleetSafetyActionIssue::UnsafeStateForAction {
                aircraft_id: aircraft.aircraft_id.clone(),
                state: item.state,
            });
            compliant = false;
        }
        for remediation in &action.required_remediation_ids {
            if !item.remediations_completed.contains(remediation) {
                issues.push(FleetSafetyActionIssue::MissingRemediation {
                    aircraft_id: aircraft.aircraft_id.clone(),
                    remediation_id: remediation.clone(),
                });
                compliant = false;
            }
        }
        if self.policy.require_evidence_for_compliance
            && item.state == FleetSafetyComplianceState::Compliant
            && item.evidence_ids.is_empty()
        {
            issues.push(FleetSafetyActionIssue::MissingComplianceArtifact {
                aircraft_id: aircraft.aircraft_id.clone(),
            });
            compliant = false;
        }
        compliant
            && (item.state == FleetSafetyComplianceState::Compliant || item.state == required_state)
    }
}

fn required_state(kind: FleetSafetyActionKind) -> FleetSafetyComplianceState {
    match kind {
        FleetSafetyActionKind::Advisory => FleetSafetyComplianceState::Notified,
        FleetSafetyActionKind::RestrictOperation => FleetSafetyComplianceState::Restricted,
        FleetSafetyActionKind::MandatoryInspection => FleetSafetyComplianceState::Grounded,
        FleetSafetyActionKind::Ground
        | FleetSafetyActionKind::RecallSoftware
        | FleetSafetyActionKind::RecallHardware => FleetSafetyComplianceState::Grounded,
    }
}

fn state_satisfies(kind: FleetSafetyActionKind, state: FleetSafetyComplianceState) -> bool {
    if state == FleetSafetyComplianceState::Compliant {
        return true;
    }
    match kind {
        FleetSafetyActionKind::Advisory => matches!(
            state,
            FleetSafetyComplianceState::Notified
                | FleetSafetyComplianceState::Restricted
                | FleetSafetyComplianceState::Grounded
                | FleetSafetyComplianceState::Remediating
        ),
        FleetSafetyActionKind::RestrictOperation => matches!(
            state,
            FleetSafetyComplianceState::Restricted
                | FleetSafetyComplianceState::Grounded
                | FleetSafetyComplianceState::Remediating
        ),
        FleetSafetyActionKind::MandatoryInspection
        | FleetSafetyActionKind::Ground
        | FleetSafetyActionKind::RecallSoftware
        | FleetSafetyActionKind::RecallHardware => matches!(
            state,
            FleetSafetyComplianceState::Grounded | FleetSafetyComplianceState::Remediating
        ),
    }
}

fn scope_matches(scope: &FleetSafetyScope, aircraft: &FleetAircraftIdentity) -> bool {
    match scope {
        FleetSafetyScope::EntireFleet => true,
        FleetSafetyScope::AircraftIds(ids) => ids.contains(&aircraft.aircraft_id),
        FleetSafetyScope::AirframeModel(model) => aircraft.airframe_model == *model,
        FleetSafetyScope::DeploymentDigest(digest) => aircraft.deployment_digest == *digest,
        FleetSafetyScope::ComponentPartNumber(part) => {
            aircraft.component_part_numbers.contains(part)
        }
        FleetSafetyScope::ComponentSerialRange {
            prefix,
            first,
            last,
        } => aircraft.component_serials.iter().any(|serial| {
            serial
                .strip_prefix(prefix)
                .and_then(|suffix| suffix.parse::<u64>().ok())
                .is_some_and(|number| number >= *first && number <= *last)
        }),
    }
}

fn issue_is_failure(issue: &FleetSafetyActionIssue) -> bool {
    !matches!(
        issue,
        FleetSafetyActionIssue::MissingAuthenticityEvidence
            | FleetSafetyActionIssue::MissingComplianceEvidence { .. }
            | FleetSafetyActionIssue::MissingComplianceArtifact { .. }
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn action() -> FleetSafetyAction {
        FleetSafetyAction {
            action_id: "FSA-1".into(),
            kind: FleetSafetyActionKind::Ground,
            scope: FleetSafetyScope::EntireFleet,
            issued_at_ms: 0,
            effective_at_ms: 0,
            compliance_deadline_ms: 1_000,
            expires_at_ms: None,
            authority_id: "safety-board".into(),
            reason_evidence_id: "incident-1".into(),
            required_remediation_ids: BTreeSet::from(["inspect-tail".into()]),
            authenticity_evidence_id: Some("sig-1".into()),
        }
    }
    fn aircraft() -> FleetAircraftIdentity {
        FleetAircraftIdentity {
            aircraft_id: "A1".into(),
            airframe_model: "H1".into(),
            deployment_digest: "d".into(),
            component_part_numbers: BTreeSet::new(),
            component_serials: BTreeSet::new(),
        }
    }
    fn compliance() -> FleetSafetyComplianceEvidence {
        FleetSafetyComplianceEvidence {
            action_id: "FSA-1".into(),
            aircraft_id: "A1".into(),
            state: FleetSafetyComplianceState::Compliant,
            acknowledged_at_ms: Some(10),
            remediations_completed: BTreeSet::from(["inspect-tail".into()]),
            evidence_ids: BTreeSet::from(["work-order".into()]),
            assessed_at_ms: 500,
        }
    }

    #[test]
    fn fully_compliant_action_passes() {
        let coordinator =
            FleetSafetyActionCoordinator::new(FleetSafetyActionPolicy::default()).unwrap();
        let report = coordinator
            .assess(&action(), &[aircraft()], &[compliance()], 500)
            .unwrap();
        assert_eq!(report.status, FleetSafetyActionStatus::Pass);
    }

    #[test]
    fn operating_after_grounding_deadline_fails() {
        let mut evidence = compliance();
        evidence.state = FleetSafetyComplianceState::Notified;
        let coordinator =
            FleetSafetyActionCoordinator::new(FleetSafetyActionPolicy::default()).unwrap();
        let report = coordinator
            .assess(&action(), &[aircraft()], &[evidence], 2_000)
            .unwrap();
        assert_eq!(report.status, FleetSafetyActionStatus::Fail);
    }

    #[test]
    fn missing_evidence_is_incomplete_before_deadline() {
        let coordinator =
            FleetSafetyActionCoordinator::new(FleetSafetyActionPolicy::default()).unwrap();
        let report = coordinator
            .assess(&action(), &[aircraft()], &[], 500)
            .unwrap();
        assert_eq!(report.status, FleetSafetyActionStatus::Incomplete);
    }
}
