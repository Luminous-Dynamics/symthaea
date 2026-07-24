// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Replay contracts for durable gateway state and cross-journal evidence.

use crate::crypto_digest::{Sha256, Sha256Digest};
use crate::gateway_state::{FabricationGatewayState, GatewayEvidenceDigests, GatewayStateError};
use crate::reconciliation::SubmissionAuditReconciliationReport;
use crate::rotation::{KeyRotationPolicy, TrustRotationViolation, digest_rotation_policy};
use serde::{Deserialize, Serialize};

pub const GATEWAY_REPLAY_SCHEMA: &str = "symthaea.fabrication.gateway-replay-contract.v3";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct GatewayReplayContract {
    pub schema_version: String,
    pub operational_replay_digest: Sha256Digest,
    pub gateway_state_digest: Sha256Digest,
    pub gateway_generation: u64,
    pub previous_gateway_state_digest: Option<Sha256Digest>,
    pub trust_snapshot_digest: Sha256Digest,
    pub audit_journal_digest: Sha256Digest,
    pub audit_head: Option<Sha256Digest>,
    pub session_tracker_digest: Sha256Digest,
    pub telemetry_tracker_digest: Sha256Digest,
    pub submission_ledger_digest: Sha256Digest,
    pub submission_head: Option<Sha256Digest>,
    pub operator_command_tracker_digest: Sha256Digest,
    pub gateway_consensus_tracker_digest: Sha256Digest,
    pub incident_ledger_digest: Sha256Digest,
    pub incident_head: Option<Sha256Digest>,
    pub reconciliation_digest: Sha256Digest,
    pub rotation_policy_digest: Option<Sha256Digest>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GatewayReplayError {
    GatewayState(GatewayStateError),
    SubmissionEvidenceNotReconciled,
    RotationPolicy(Vec<TrustRotationViolation>),
    Encoding(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GatewayReplayMismatch {
    SchemaVersion,
    OperationalReplay,
    GatewayState,
    GatewayGeneration,
    PreviousState,
    TrustSnapshot,
    AuditJournal,
    AuditHead,
    SessionTracker,
    TelemetryTracker,
    SubmissionLedger,
    SubmissionHead,
    OperatorCommandTracker,
    GatewayConsensusTracker,
    IncidentLedger,
    IncidentHead,
    Reconciliation,
    RotationPolicy,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GatewayReplayVerificationReport {
    pub mismatches: Vec<GatewayReplayMismatch>,
}

impl GatewayReplayVerificationReport {
    pub fn reproducible(&self) -> bool {
        self.mismatches.is_empty()
    }
}

pub fn build_gateway_replay_contract(
    state: &FabricationGatewayState,
    operational_replay_digest: Sha256Digest,
    reconciliation: &SubmissionAuditReconciliationReport,
    rotation_policy: Option<&KeyRotationPolicy>,
) -> Result<GatewayReplayContract, GatewayReplayError> {
    if !reconciliation.reconciled() {
        return Err(GatewayReplayError::SubmissionEvidenceNotReconciled);
    }
    let gateway_state_digest = state.digest().map_err(GatewayReplayError::GatewayState)?;
    let evidence = state
        .evidence_digests()
        .map_err(GatewayReplayError::GatewayState)?;
    let reconciliation_digest = digest_reconciliation_report(reconciliation)?;
    let rotation_policy_digest = rotation_policy
        .map(digest_rotation_policy)
        .transpose()
        .map_err(GatewayReplayError::RotationPolicy)?;
    Ok(contract_from_evidence(
        state,
        operational_replay_digest,
        gateway_state_digest,
        evidence,
        reconciliation_digest,
        rotation_policy_digest,
    ))
}

pub fn verify_gateway_replay_contract(
    contract: &GatewayReplayContract,
    state: &FabricationGatewayState,
    operational_replay_digest: Sha256Digest,
    reconciliation: &SubmissionAuditReconciliationReport,
    rotation_policy: Option<&KeyRotationPolicy>,
) -> Result<GatewayReplayVerificationReport, GatewayReplayError> {
    let expected = build_gateway_replay_contract(
        state,
        operational_replay_digest,
        reconciliation,
        rotation_policy,
    )?;
    let mut mismatches = Vec::new();
    if contract.schema_version != GATEWAY_REPLAY_SCHEMA {
        mismatches.push(GatewayReplayMismatch::SchemaVersion);
    }
    compare(
        contract.operational_replay_digest != expected.operational_replay_digest,
        GatewayReplayMismatch::OperationalReplay,
        &mut mismatches,
    );
    compare(
        contract.gateway_state_digest != expected.gateway_state_digest,
        GatewayReplayMismatch::GatewayState,
        &mut mismatches,
    );
    compare(
        contract.gateway_generation != expected.gateway_generation,
        GatewayReplayMismatch::GatewayGeneration,
        &mut mismatches,
    );
    compare(
        contract.previous_gateway_state_digest != expected.previous_gateway_state_digest,
        GatewayReplayMismatch::PreviousState,
        &mut mismatches,
    );
    compare(
        contract.trust_snapshot_digest != expected.trust_snapshot_digest,
        GatewayReplayMismatch::TrustSnapshot,
        &mut mismatches,
    );
    compare(
        contract.audit_journal_digest != expected.audit_journal_digest,
        GatewayReplayMismatch::AuditJournal,
        &mut mismatches,
    );
    compare(
        contract.audit_head != expected.audit_head,
        GatewayReplayMismatch::AuditHead,
        &mut mismatches,
    );
    compare(
        contract.session_tracker_digest != expected.session_tracker_digest,
        GatewayReplayMismatch::SessionTracker,
        &mut mismatches,
    );
    compare(
        contract.telemetry_tracker_digest != expected.telemetry_tracker_digest,
        GatewayReplayMismatch::TelemetryTracker,
        &mut mismatches,
    );
    compare(
        contract.submission_ledger_digest != expected.submission_ledger_digest,
        GatewayReplayMismatch::SubmissionLedger,
        &mut mismatches,
    );
    compare(
        contract.submission_head != expected.submission_head,
        GatewayReplayMismatch::SubmissionHead,
        &mut mismatches,
    );
    compare(
        contract.operator_command_tracker_digest != expected.operator_command_tracker_digest,
        GatewayReplayMismatch::OperatorCommandTracker,
        &mut mismatches,
    );
    compare(
        contract.gateway_consensus_tracker_digest != expected.gateway_consensus_tracker_digest,
        GatewayReplayMismatch::GatewayConsensusTracker,
        &mut mismatches,
    );
    compare(
        contract.incident_ledger_digest != expected.incident_ledger_digest,
        GatewayReplayMismatch::IncidentLedger,
        &mut mismatches,
    );
    compare(
        contract.incident_head != expected.incident_head,
        GatewayReplayMismatch::IncidentHead,
        &mut mismatches,
    );
    compare(
        contract.reconciliation_digest != expected.reconciliation_digest,
        GatewayReplayMismatch::Reconciliation,
        &mut mismatches,
    );
    compare(
        contract.rotation_policy_digest != expected.rotation_policy_digest,
        GatewayReplayMismatch::RotationPolicy,
        &mut mismatches,
    );
    Ok(GatewayReplayVerificationReport { mismatches })
}

pub fn digest_gateway_replay_contract(
    contract: &GatewayReplayContract,
) -> Result<Sha256Digest, GatewayReplayError> {
    let bytes = serde_json::to_vec(contract)
        .map_err(|error| GatewayReplayError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.gateway-replay-contract-digest.v1\0");
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

pub fn digest_reconciliation_report(
    report: &SubmissionAuditReconciliationReport,
) -> Result<Sha256Digest, GatewayReplayError> {
    if !report.reconciled() {
        return Err(GatewayReplayError::SubmissionEvidenceNotReconciled);
    }
    let mut hasher = Sha256::new();
    hasher.update(b"symthaea.fabrication.submission-reconciliation.v1\0");
    hasher.update(&(report.submission_event_count as u64).to_le_bytes());
    hasher.update(&(report.matched_event_count as u64).to_le_bytes());
    Ok(hasher.finalize())
}

fn contract_from_evidence(
    state: &FabricationGatewayState,
    operational_replay_digest: Sha256Digest,
    gateway_state_digest: Sha256Digest,
    evidence: GatewayEvidenceDigests,
    reconciliation_digest: Sha256Digest,
    rotation_policy_digest: Option<Sha256Digest>,
) -> GatewayReplayContract {
    GatewayReplayContract {
        schema_version: GATEWAY_REPLAY_SCHEMA.into(),
        operational_replay_digest,
        gateway_state_digest,
        gateway_generation: state.generation,
        previous_gateway_state_digest: state.previous_state_digest,
        trust_snapshot_digest: evidence.trust_snapshot,
        audit_journal_digest: evidence.audit_journal,
        audit_head: evidence.audit_head,
        session_tracker_digest: evidence.session_tracker,
        telemetry_tracker_digest: evidence.telemetry_tracker,
        submission_ledger_digest: evidence.submission_ledger,
        submission_head: evidence.submission_head,
        operator_command_tracker_digest: evidence.operator_command_tracker,
        gateway_consensus_tracker_digest: evidence.gateway_consensus_tracker,
        incident_ledger_digest: evidence.incident_ledger,
        incident_head: evidence.incident_head,
        reconciliation_digest,
        rotation_policy_digest,
    }
}

fn compare(
    differs: bool,
    mismatch: GatewayReplayMismatch,
    output: &mut Vec<GatewayReplayMismatch>,
) {
    if differs {
        output.push(mismatch);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attestation::SignatureAlgorithm;
    use crate::audit::AuditJournal;
    use crate::gateway_consensus_tracker::GatewayConsensusTracker;
    use crate::gateway_state::FabricationGatewayState;
    use crate::incident_ledger::IncidentLedger;
    use crate::operator_command_tracker::OperatorCommandTracker;
    use crate::session::MachineSessionTracker;
    use crate::submission_ledger::SubmissionLedger;
    use crate::telemetry_tracker::MachineTelemetryTracker;
    use crate::trust::{KeyLifecycleStatus, KeyTrustRecord, KeyUsage, TrustSnapshot};
    use std::collections::BTreeSet;

    fn state() -> FabricationGatewayState {
        let trust = TrustSnapshot::new(
            1,
            100,
            1_000,
            vec![KeyTrustRecord {
                algorithm: SignatureAlgorithm::Ed25519,
                key_id: "root".into(),
                not_before_unix_s: 100,
                not_after_unix_s: None,
                status: KeyLifecycleStatus::Active,
                usages: BTreeSet::from([KeyUsage::FabricationManifest]),
            }],
        )
        .unwrap();
        FabricationGatewayState::genesis(
            500_000,
            trust,
            AuditJournal::default(),
            MachineSessionTracker::default(),
            MachineTelemetryTracker::default(),
            SubmissionLedger::default(),
            OperatorCommandTracker::default(),
            GatewayConsensusTracker::default(),
            IncidentLedger::default(),
        )
        .unwrap()
    }

    fn reconciliation() -> SubmissionAuditReconciliationReport {
        SubmissionAuditReconciliationReport {
            submission_event_count: 0,
            matched_event_count: 0,
            mismatches: Vec::new(),
        }
    }

    #[test]
    fn state_or_policy_drift_is_reported() {
        let state = state();
        let operational = Sha256Digest([3; 32]);
        let policy = KeyRotationPolicy::default();
        let contract =
            build_gateway_replay_contract(&state, operational, &reconciliation(), Some(&policy))
                .unwrap();
        let mut changed = policy.clone();
        changed.minimum_overlap_s += 1;
        let report = verify_gateway_replay_contract(
            &contract,
            &state,
            operational,
            &reconciliation(),
            Some(&changed),
        )
        .unwrap();
        assert_eq!(
            report.mismatches,
            vec![GatewayReplayMismatch::RotationPolicy]
        );
    }

    #[test]
    fn unreconciled_evidence_cannot_enter_replay() {
        let mut report = reconciliation();
        report.submission_event_count = 1;
        assert!(matches!(
            build_gateway_replay_contract(&state(), Sha256Digest([3; 32]), &report, None,),
            Err(GatewayReplayError::SubmissionEvidenceNotReconciled)
        ));
    }
}
