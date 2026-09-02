// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Deterministic acceptance contracts for operator authority, update safety,
//! audit continuity, degraded operation and checkpoint recovery.

use crate::audit_chain::{AuditChainError, AuditEvent, AuditLedger, DeterministicAuditDigest};
use crate::degraded_operations::{
    DegradedMode, DegradedObservation, DegradedOperationsSupervisor, DegradedPolicy,
};
use crate::embodiment::SubterraneanEmbodiment;
use crate::operator_authority::{
    OperatorAuthority, OperatorAuthorityRejection, OperatorConstraint, OperatorDecision,
};
use crate::operator_authority::recovery_authority::{
    RecoveryApprovalEnvelopeV1, RecoveryDigest, RecoveryProposalV1,
};
use crate::operator_protocol::{
    AuthenticationLevel, OperatorCommand, OperatorCommandEnvelope, OperatorId, OperatorRole,
};
use crate::recovery_journal::{DeterministicJournalDigest, RecoveryJournal};
use crate::update_control::{
    ArtifactDigest, UPDATE_MANIFEST_SCHEMA_VERSION, UpdateManager, UpdateManifest,
    UpdatePreconditions, UpdateState,
};
use serde::{Deserialize, Serialize};
use symthaea_core::genesis::GenesisSeed;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuthorityContract {
    ReplayResistance,
    IndependentRecoveryQuorum,
    HazardBlocksResume,
    RestrictionMonotonicity,
    RecoveryApprovalInvalidation,
    RecoveryProposalBinding,
    CheckpointPreservesRestriction,
    AuditChainContinuity,
    UpdateRollback,
    WatchdogRecoveryLock,
    JournalFallback,
}

impl AuthorityContract {
    pub const ALL: [Self; 11] = [
        Self::ReplayResistance,
        Self::IndependentRecoveryQuorum,
        Self::HazardBlocksResume,
        Self::RestrictionMonotonicity,
        Self::RecoveryApprovalInvalidation,
        Self::RecoveryProposalBinding,
        Self::CheckpointPreservesRestriction,
        Self::AuditChainContinuity,
        Self::UpdateRollback,
        Self::WatchdogRecoveryLock,
        Self::JournalFallback,
    ];

    pub const fn label(self) -> &'static str {
        match self {
            Self::ReplayResistance => "replay_resistance",
            Self::IndependentRecoveryQuorum => "independent_recovery_quorum",
            Self::HazardBlocksResume => "hazard_blocks_resume",
            Self::RestrictionMonotonicity => "restriction_monotonicity",
            Self::RecoveryApprovalInvalidation => "recovery_approval_invalidation",
            Self::RecoveryProposalBinding => "recovery_proposal_binding",
            Self::CheckpointPreservesRestriction => "checkpoint_preserves_restriction",
            Self::AuditChainContinuity => "audit_chain_continuity",
            Self::UpdateRollback => "update_rollback",
            Self::WatchdogRecoveryLock => "watchdog_recovery_lock",
            Self::JournalFallback => "journal_fallback",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AuthorityGateFailure {
    pub contract: AuthorityContract,
    pub detail: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AuthorityValidationReport {
    pub passed: Vec<AuthorityContract>,
    pub failures: Vec<AuthorityGateFailure>,
}

impl AuthorityValidationReport {
    pub fn is_success(&self) -> bool {
        self.failures.is_empty() && self.passed.len() == AuthorityContract::ALL.len()
    }

    pub fn to_pretty_json(&self) -> Result<String, serde_json::Error> {
        serde_json::to_string_pretty(self)
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct AuthorityValidator;

impl AuthorityValidator {
    fn envelope(operator: u64, sequence: u64, proposal_id: u64, command: OperatorCommand) -> OperatorCommandEnvelope {
        OperatorCommandEnvelope {
            operator: OperatorId(operator),
            role: OperatorRole::SafetyOfficer,
            authentication: AuthenticationLevel::HardwareBacked,
            epoch: 1,
            sequence,
            proposal_id,
            issued_step: 10,
            expires_step: 1_000,
            command,
        }
    }

    fn recovery_proposal(proposal_id: u64, active: OperatorConstraint) -> RecoveryProposalV1 {
        RecoveryProposalV1::new(
            proposal_id,
            active,
            RecoveryDigest([11; 32]),
            RecoveryDigest([12; 32]),
            RecoveryDigest([13; 32]),
            1,
            1,
            10,
            1_000,
        )
    }

    fn recovery_approval(operator: u64, sequence: u64, proposal: RecoveryProposalV1) -> RecoveryApprovalEnvelopeV1 {
        RecoveryApprovalEnvelopeV1 {
            operator: OperatorId(operator),
            role: OperatorRole::SafetyOfficer,
            authentication: AuthenticationLevel::HardwareBacked,
            epoch: 1,
            sequence,
            approval_issued_step: 20,
            proposal,
        }
    }

    fn digest(byte: u8) -> ArtifactDigest { ArtifactDigest([byte; 32]) }

    fn update_manifest() -> UpdateManifest {
        UpdateManifest {
            schema_version: UPDATE_MANIFEST_SCHEMA_VERSION,
            release_id: 2,
            artifact_digest: Self::digest(2),
            configuration_digest: Self::digest(3),
            rollback_digest: Self::digest(1),
            minimum_checkpoint_schema: 1,
            issued_epoch: 2,
            expires_step: 1_000,
        }
    }

    fn update_preconditions() -> UpdatePreconditions {
        UpdatePreconditions {
            at_surface_or_service_bay: true,
            active_work: false,
            physical_hazard_clear: true,
            battery_ratio: 0.9,
            operator_constraint: OperatorConstraint::MaintenanceLock,
        }
    }

    fn evaluate(contract: AuthorityContract) -> Result<(), String> {
        match contract {
            AuthorityContract::ReplayResistance => {
                let mut authority = OperatorAuthority::default();
                let command = Self::envelope(1, 1, 1, OperatorCommand::HoldPosition);
                authority.ingest(command, 20, true).map_err(|error| format!("initial command rejected: {error:?}"))?;
                if authority.ingest(command, 21, true) != Err(OperatorAuthorityRejection::Replay) {
                    return Err("replayed sequence was not rejected".to_string());
                }
                Ok(())
            }
            AuthorityContract::IndependentRecoveryQuorum => {
                let mut authority = OperatorAuthority::default();
                authority.ingest(Self::envelope(1, 1, 1, OperatorCommand::EmergencyStop), 20, true)
                    .map_err(|error| format!("stop rejected: {error:?}"))?;
                let proposal = Self::recovery_proposal(9, OperatorConstraint::EmergencyStop);
                let first = authority.approve_recovery(Self::recovery_approval(1, 2, proposal), 21)
                    .map_err(|error| format!("first approval rejected: {error:?}"))?;
                if !matches!(first, OperatorDecision::PendingQuorum { approvals: 1, .. })
                    || authority.constraint() != OperatorConstraint::EmergencyStop
                {
                    return Err("one operator cleared a restrictive constraint".to_string());
                }
                let second = authority.approve_recovery(Self::recovery_approval(2, 1, proposal), 22)
                    .map_err(|error| format!("second approval rejected: {error:?}"))?;
                if second != OperatorDecision::Cleared || authority.constraint() != OperatorConstraint::None {
                    return Err("two approvals to the exact proposal did not clear recovery".to_string());
                }
                Ok(())
            }
            AuthorityContract::HazardBlocksResume => {
                let mut authority = OperatorAuthority::default();
                authority.ingest(Self::envelope(1, 1, 1, OperatorCommand::EmergencyStop), 20, true)
                    .map_err(|error| format!("stop rejected: {error:?}"))?;
                let result = authority.ingest(Self::envelope(1, 2, 9, OperatorCommand::ResumeNominal), 21, false);
                if result != Err(OperatorAuthorityRejection::RecoveryProposalRequired)
                    || authority.constraint() != OperatorConstraint::EmergencyStop
                {
                    return Err("legacy hazard boolean still authorized resume".to_string());
                }
                Ok(())
            }
            AuthorityContract::RestrictionMonotonicity => {
                let mut authority = OperatorAuthority::default();
                authority.ingest(Self::envelope(1, 1, 1, OperatorCommand::EmergencyStop), 20, true)
                    .map_err(|error| format!("stop rejected: {error:?}"))?;
                let result = authority.ingest(Self::envelope(1, 2, 2, OperatorCommand::ReturnHome), 21, true);
                if result != Err(OperatorAuthorityRejection::WouldWeakenActiveConstraint)
                    || authority.constraint() != OperatorConstraint::EmergencyStop
                {
                    return Err("ordinary command weakened emergency stop".to_string());
                }
                Ok(())
            }
            AuthorityContract::RecoveryApprovalInvalidation => {
                let mut authority = OperatorAuthority::default();
                authority.ingest(Self::envelope(1, 1, 1, OperatorCommand::HoldPosition), 20, true)
                    .map_err(|error| format!("hold rejected: {error:?}"))?;
                let proposal = Self::recovery_proposal(9, OperatorConstraint::HoldPosition);
                let first = authority.approve_recovery(Self::recovery_approval(1, 2, proposal), 21)
                    .map_err(|error| format!("first approval rejected: {error:?}"))?;
                if !matches!(first, OperatorDecision::PendingQuorum { approvals: 1, .. })
                    || authority.pending_approvals(9) != 1
                {
                    return Err("first recovery approval was not retained".to_string());
                }
                authority.ingest(Self::envelope(2, 1, 2, OperatorCommand::EmergencyStop), 22, true)
                    .map_err(|error| format!("new stop rejected: {error:?}"))?;
                if authority.pending_approvals(9) != 0 {
                    return Err("new restriction retained stale recovery approvals".to_string());
                }
                let result = authority.approve_recovery(Self::recovery_approval(2, 2, proposal), 23);
                if !matches!(result, Err(OperatorAuthorityRejection::RecoveryProposal(_)))
                    || authority.constraint() != OperatorConstraint::EmergencyStop
                {
                    return Err("proposal for earlier restriction survived the new stop".to_string());
                }
                Ok(())
            }
            AuthorityContract::RecoveryProposalBinding => {
                let mut authority = OperatorAuthority::default();
                authority.ingest(Self::envelope(1, 1, 1, OperatorCommand::HoldPosition), 20, true)
                    .map_err(|error| format!("hold rejected: {error:?}"))?;
                let proposal = Self::recovery_proposal(9, OperatorConstraint::HoldPosition);
                authority.approve_recovery(Self::recovery_approval(1, 2, proposal), 21)
                    .map_err(|error| format!("first approval rejected: {error:?}"))?;
                let changed = RecoveryProposalV1::new(
                    9,
                    OperatorConstraint::HoldPosition,
                    RecoveryDigest([11; 32]),
                    RecoveryDigest([99; 32]),
                    RecoveryDigest([13; 32]),
                    1,
                    1,
                    10,
                    1_000,
                );
                if authority.approve_recovery(Self::recovery_approval(2, 1, changed), 22)
                    != Err(OperatorAuthorityRejection::ConflictingProposal)
                {
                    return Err("changed evidence joined an existing recovery quorum".to_string());
                }
                Ok(())
            }
            AuthorityContract::CheckpointPreservesRestriction => {
                let genesis = GenesisSeed::from_phrase("authority restriction checkpoint");
                let mut source = SubterraneanEmbodiment::new(&genesis);
                source.ingest_operator_command(Self::envelope(1, 1, 1, OperatorCommand::HoldPosition))
                    .map_err(|error| format!("source hold rejected: {error:?}"))?;
                let checkpoint = source.operational_checkpoint();
                let mut restored = SubterraneanEmbodiment::new(&genesis);
                restored.load_operational_checkpoint(&checkpoint)
                    .map_err(|error| format!("checkpoint restore failed: {error:?}"))?;
                if restored.operator_constraint() != OperatorConstraint::HoldPosition {
                    return Err("checkpoint restore widened operator authority".to_string());
                }
                Ok(())
            }
            AuthorityContract::AuditChainContinuity => {
                let provider = DeterministicAuditDigest;
                let mut ledger = AuditLedger::new(8, Self::digest(7));
                ledger.append(&provider, AuditEvent::OperatorCommand {
                    operator_id: 1,
                    proposal_id: 2,
                    command_code: OperatorCommand::HoldPosition.code(),
                    accepted: true,
                });
                ledger.append(&provider, AuditEvent::OperatorConstraint { constraint_code: OperatorConstraint::HoldPosition.code() });
                if ledger.verify(&provider) != Ok(()) {
                    return Err("valid audit chain did not verify".to_string());
                }
                let mut records = ledger.records();
                records[0].event = AuditEvent::OperatorCommand {
                    operator_id: 99,
                    proposal_id: 2,
                    command_code: OperatorCommand::HoldPosition.code(),
                    accepted: true,
                };
                if AuditLedger::verify_records(&provider, &records, ledger.chain_head()) != Err(AuditChainError::DigestMismatch) {
                    return Err("modified audit record was not detected".to_string());
                }
                Ok(())
            }
            AuthorityContract::UpdateRollback => {
                let mut manager = UpdateManager::new(Self::digest(1), 1).map_err(|error| format!("manager init failed: {error:?}"))?;
                manager.stage(Self::update_manifest(), 10, 1, Self::update_preconditions()).map_err(|error| format!("stage failed: {error:?}"))?;
                manager.activate(11, 50, Self::update_preconditions()).map_err(|error| format!("activation failed: {error:?}"))?;
                let state = manager.observe_health(false, 12).map_err(|error| format!("health observation failed: {error:?}"))?;
                if state != UpdateState::RollbackRequired {
                    return Err("failed health did not require rollback".to_string());
                }
                if manager.rollback().map_err(|error| format!("rollback failed: {error:?}"))? != Self::digest(1) {
                    return Err("rollback did not restore previous digest".to_string());
                }
                Ok(())
            }
            AuthorityContract::WatchdogRecoveryLock => {
                let mut supervisor = DegradedOperationsSupervisor::new(DegradedPolicy { watchdog_failure_limit: 2, ..Default::default() });
                let observation = DegradedObservation {
                    operator_link_fresh: true,
                    control_loop_healthy: false,
                    checkpoint_valid: true,
                    reboot_count_in_window: 0,
                    battery_ratio: 0.8,
                    return_feasible: true,
                    at_surface_or_service_bay: false,
                };
                supervisor.update(observation);
                if supervisor.update(observation).current != DegradedMode::RecoveryRequired {
                    return Err("watchdog failures did not latch recovery lock".to_string());
                }
                Ok(())
            }
            AuthorityContract::JournalFallback => {
                let genesis = GenesisSeed::from_phrase("authority journal validation");
                let embodiment = SubterraneanEmbodiment::new(&genesis);
                let checkpoint = embodiment.operational_checkpoint();
                let good = DeterministicJournalDigest;
                struct DifferentDigest;
                impl crate::recovery_journal::JournalDigestProvider for DifferentDigest {
                    fn digest(&self, generation: u64, checkpoint: &crate::SubterraneanOperationalCheckpoint) -> ArtifactDigest {
                        let mut digest = DeterministicJournalDigest.digest(generation, checkpoint);
                        digest.0[0] ^= 0x55;
                        digest
                    }
                }
                let mut journal = RecoveryJournal::new();
                journal.write(&good, 1, checkpoint.clone()).map_err(|error| format!("first write failed: {error:?}"))?;
                journal.write(&DifferentDigest, 2, checkpoint).map_err(|error| format!("second write failed: {error:?}"))?;
                let restored = journal.latest_valid(&good).map_err(|error| format!("journal recovery failed: {error:?}"))?;
                if restored.generation != 1 {
                    return Err("journal did not fall back to older valid slot".to_string());
                }
                Ok(())
            }
        }
    }

    pub fn run() -> AuthorityValidationReport {
        let mut passed = Vec::new();
        let mut failures = Vec::new();
        for contract in AuthorityContract::ALL {
            match Self::evaluate(contract) {
                Ok(()) => passed.push(contract),
                Err(detail) => failures.push(AuthorityGateFailure { contract, detail }),
            }
        }
        AuthorityValidationReport { passed, failures }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reference_authority_contracts_pass() {
        let report = AuthorityValidator::run();
        assert!(report.is_success(), "failures: {:?}", report.failures);
    }
}
