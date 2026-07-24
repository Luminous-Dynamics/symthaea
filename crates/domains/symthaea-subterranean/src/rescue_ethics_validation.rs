// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic release contracts for human-rescue ethics assurance.

use crate::operator_protocol::{AuthenticationLevel, OperatorId};
use crate::rescue::{RescueCaseId, RescueHandoffState};
use crate::rescue_consent::{
    RESCUE_CONSENT_SCHEMA_VERSION, RescueConsentDecision, RescueConsentDisposition,
    RescueConsentLedger, RescueConsentRejection, RescueConsentStatement,
};
use crate::rescue_emergency_authority::{
    EMERGENCY_RESCUE_AUTHORITY_SCHEMA_VERSION, EmergencyRescueApproval,
    EmergencyRescueAuthorization, EmergencyRescueAuthorityLedger, EmergencyRescueRole,
};
use crate::rescue_ethics::{RescueEthicsAuthority, RescueEthicsSupervisor};
use crate::rescue_subject_claim::{
    RESCUE_SUBJECT_CLAIM_SCHEMA_VERSION, RescueCareUrgency, RescueSubjectClaim,
    RescueSubjectClaimAssessment, RescueSubjectClaimLedger,
};
use crate::rescue_triage::{
    RESCUE_TRIAGE_SCHEMA_VERSION, RescueTriageCandidate, RescueTriageSupervisor,
};
use crate::team::AgentId;
use crate::team_operations::TeamCoordinator;
use crate::types::{CUTTER_HEAD, LEFT_TRACK, SubterraneanCommand};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RescueEthicsContract {
    ConsentReplayRejected,
    WithdrawalStopsActiveRescue,
    EmergencyAuthorityRequiresIndependentRoles,
    ConflictingIdentityRequiresReconciliation,
    RefusalDominatesUrgency,
    TriageExcludesProtectedAttributes,
    RecoveryActuatorsSurviveEthicsHold,
    CheckpointPreservesConsentAuthority,
}

impl RescueEthicsContract {
    pub const ALL: [Self; 8] = [
        Self::ConsentReplayRejected,
        Self::WithdrawalStopsActiveRescue,
        Self::EmergencyAuthorityRequiresIndependentRoles,
        Self::ConflictingIdentityRequiresReconciliation,
        Self::RefusalDominatesUrgency,
        Self::TriageExcludesProtectedAttributes,
        Self::RecoveryActuatorsSurviveEthicsHold,
        Self::CheckpointPreservesConsentAuthority,
    ];
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RescueEthicsGateFailure {
    pub contract: RescueEthicsContract,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RescueEthicsValidationReport {
    pub passed: Vec<RescueEthicsContract>,
    pub failures: Vec<RescueEthicsGateFailure>,
}

impl RescueEthicsValidationReport {
    pub fn passes(&self) -> bool {
        self.failures.is_empty() && self.passed.len() == RescueEthicsContract::ALL.len()
    }
}

#[derive(Debug, Default, Clone, Copy)]
pub struct RescueEthicsValidator;

impl RescueEthicsValidator {
    pub fn validate(self) -> RescueEthicsValidationReport {
        let mut passed = Vec::new();
        let mut failures = Vec::new();
        for contract in RescueEthicsContract::ALL {
            match evaluate(contract) {
                Ok(()) => passed.push(contract),
                Err(detail) => failures.push(RescueEthicsGateFailure { contract, detail }),
            }
        }
        RescueEthicsValidationReport { passed, failures }
    }
}

fn consent_statement(sequence: u64, decision: RescueConsentDecision) -> RescueConsentStatement {
    RescueConsentStatement {
        schema_version: RESCUE_CONSENT_SCHEMA_VERSION,
        subject: AgentId::new(2),
        case_id: RescueCaseId(7),
        epoch: 1,
        sequence,
        issued_step: 10 + sequence,
        expires_step: 100,
        decision,
    }
}

fn emergency_authorization(distinct: bool) -> EmergencyRescueAuthorization {
    EmergencyRescueAuthorization {
        schema_version: EMERGENCY_RESCUE_AUTHORITY_SCHEMA_VERSION,
        authorization_id: 9,
        subject: AgentId::new(2),
        case_id: RescueCaseId(7),
        epoch: 1,
        sequence: 1,
        issued_step: 10,
        expires_step: 100,
        immediate_threat: true,
        communication_unavailable: true,
        approvals: vec![
            EmergencyRescueApproval {
                operator: OperatorId::new(11),
                role: EmergencyRescueRole::SafetyOfficer,
                authentication: AuthenticationLevel::HardwareBacked,
            },
            EmergencyRescueApproval {
                operator: OperatorId::new(if distinct { 12 } else { 11 }),
                role: EmergencyRescueRole::IndependentWitness,
                authentication: AuthenticationLevel::HardwareBacked,
            },
        ],
    }
}

fn subject_claim(reporter: u64, binding: u64) -> RescueSubjectClaim {
    RescueSubjectClaim {
        schema_version: RESCUE_SUBJECT_CLAIM_SCHEMA_VERSION,
        case_id: RescueCaseId(7),
        subject: AgentId::new(2),
        reporter: AgentId::new(reporter),
        identity_binding: binding,
        care_urgency: RescueCareUrgency::Critical,
        communication_unavailable: true,
        epoch: 1,
        sequence: 1,
        issued_step: 10,
        expires_step: 100,
    }
}

fn candidate(
    subject: u64,
    consent: RescueConsentDisposition,
    claim_assessment: RescueSubjectClaimAssessment,
) -> RescueTriageCandidate {
    RescueTriageCandidate {
        schema_version: RESCUE_TRIAGE_SCHEMA_VERSION,
        subject: AgentId::new(subject),
        case_id: RescueCaseId(7),
        observed_step: 10,
        hazard_severity: 0.95,
        survival_window_steps: 20,
        route_reachable: true,
        rescue_energy_ratio: 0.2,
        evidence_confidence: 0.9,
        consent,
        claim_assessment,
        emergency_authorized: false,
    }
}

fn clear_claim() -> RescueSubjectClaimAssessment {
    RescueSubjectClaimAssessment {
        case_id: RescueCaseId(7),
        subject: Some(AgentId::new(2)),
        trusted_reporters: 2,
        identity_bindings: 1,
        identity_conflict: false,
        care_urgency: RescueCareUrgency::Critical,
        care_conflict: false,
        communication_unavailable_corroborated: false,
        reasons: Vec::new(),
    }
}

fn evaluate(contract: RescueEthicsContract) -> Result<(), String> {
    match contract {
        RescueEthicsContract::ConsentReplayRejected => {
            let mut ledger = RescueConsentLedger::new();
            let statement = consent_statement(1, RescueConsentDecision::Consent);
            ledger
                .ingest(statement, 20)
                .map_err(|error| format!("{error:?}"))?;
            (ledger.ingest(statement, 21) == Err(RescueConsentRejection::Replay))
                .then_some(())
                .ok_or_else(|| "replayed consent statement was accepted".to_string())
        }
        RescueEthicsContract::WithdrawalStopsActiveRescue => {
            let mut triage = RescueTriageSupervisor::default();
            let assessment = triage.assess(
                20,
                &[candidate(
                    2,
                    RescueConsentDisposition::Withdrawn,
                    clear_claim(),
                )],
            );
            let ethics = RescueEthicsSupervisor::default()
                .assess(RescueHandoffState::Active, &assessment);
            (ethics.authority == RescueEthicsAuthority::HoldForReview)
                .then_some(())
                .ok_or_else(|| "withdrawn consent did not stop active rescue".to_string())
        }
        RescueEthicsContract::EmergencyAuthorityRequiresIndependentRoles => {
            let mut ledger = EmergencyRescueAuthorityLedger::new();
            let invalid = ledger.ingest(emergency_authorization(false), 20);
            let valid = ledger.ingest(emergency_authorization(true), 20);
            (invalid.is_err() && valid.is_ok())
                .then_some(())
                .ok_or_else(|| "emergency intervention did not enforce split-role authority".to_string())
        }
        RescueEthicsContract::ConflictingIdentityRequiresReconciliation => {
            let mut claims = RescueSubjectClaimLedger::new();
            claims
                .ingest(subject_claim(3, 10), true)
                .map_err(|error| format!("{error:?}"))?;
            claims
                .ingest(subject_claim(4, 11), true)
                .map_err(|error| format!("{error:?}"))?;
            let claim_assessment = claims.assess(RescueCaseId(7), 20);
            let mut triage = RescueTriageSupervisor::default();
            let triage = triage.assess(
                20,
                &[candidate(
                    2,
                    RescueConsentDisposition::Consented,
                    claim_assessment,
                )],
            );
            let ethics = RescueEthicsSupervisor::default()
                .assess(RescueHandoffState::Offered, &triage);
            (ethics.authority == RescueEthicsAuthority::ReconcileClaims)
                .then_some(())
                .ok_or_else(|| "identity conflict did not require reconciliation".to_string())
        }
        RescueEthicsContract::RefusalDominatesUrgency => {
            let mut triage = RescueTriageSupervisor::default();
            let assessment = triage.assess(
                20,
                &[
                    candidate(2, RescueConsentDisposition::Refused, clear_claim()),
                    candidate(3, RescueConsentDisposition::Consented, RescueSubjectClaimAssessment {
                        case_id: RescueCaseId(7),
                        subject: Some(AgentId::new(3)),
                        ..clear_claim()
                    }),
                ],
            );
            (assessment.selected.is_some_and(|value| value.subject == AgentId::new(3)))
                .then_some(())
                .ok_or_else(|| "urgent refusal was treated as rescue consent".to_string())
        }
        RescueEthicsContract::TriageExcludesProtectedAttributes => {
            let mut triage = RescueTriageSupervisor::default();
            let assessment = triage.assess(
                20,
                &[
                    candidate(7, RescueConsentDisposition::Consented, clear_claim()),
                    candidate(4, RescueConsentDisposition::Consented, RescueSubjectClaimAssessment {
                        case_id: RescueCaseId(7),
                        subject: Some(AgentId::new(4)),
                        ..clear_claim()
                    }),
                ],
            );
            (assessment.non_discrimination_invariant_satisfied
                && assessment.selected.is_some_and(|value| value.subject == AgentId::new(4)))
                .then_some(())
                .ok_or_else(|| "role-neutral deterministic tie-break failed".to_string())
        }
        RescueEthicsContract::RecoveryActuatorsSurviveEthicsHold => {
            let assessment = crate::rescue_ethics::RescueEthicsAssessment {
                authority: RescueEthicsAuthority::HoldForReview,
                selected_subject: Some(AgentId::new(2)),
                selected_case_id: Some(7),
                consent: RescueConsentDisposition::Withdrawn,
                emergency_authorized: false,
                identity_conflict: false,
                care_conflict: false,
                non_discrimination_invariant_satisfied: true,
                triage_candidates: 1,
                reasons: vec!["withdrawn consent".to_string()],
            };
            let mut command = SubterraneanCommand::zero();
            command.torques[CUTTER_HEAD] = 0.8;
            command.torques[LEFT_TRACK] = 0.5;
            command.dewatering_pump = 0.7;
            let constrained = assessment.constrain_command(command);
            (constrained.torques[CUTTER_HEAD] == 0.0
                && constrained.torques[LEFT_TRACK] == 0.0
                && constrained.dewatering_pump == 0.7)
                .then_some(())
                .ok_or_else(|| "ethics hold removed required recovery authority".to_string())
        }
        RescueEthicsContract::CheckpointPreservesConsentAuthority => {
            let mut coordinator = TeamCoordinator::new(AgentId::new(1));
            coordinator
                .ingest_rescue_consent(
                    consent_statement(1, RescueConsentDecision::Withdraw),
                    20,
                )
                .map_err(|error| format!("{error:?}"))?;
            let checkpoint = coordinator.recovery_checkpoint();
            let preserved = checkpoint.rescue_consent.disposition(
                AgentId::new(2),
                RescueCaseId(7),
                21,
                RescueHandoffState::Active,
            );
            (checkpoint.validate() && preserved == RescueConsentDisposition::Withdrawn)
                .then_some(())
                .ok_or_else(|| "checkpoint lost case-specific withdrawal authority".to_string())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_rescue_ethics_contracts_pass() {
        let report = RescueEthicsValidator.validate();
        assert!(report.passes(), "{:?}", report.failures);
    }
}
