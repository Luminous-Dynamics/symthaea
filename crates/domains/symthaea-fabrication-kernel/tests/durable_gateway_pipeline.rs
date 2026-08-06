// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::collections::BTreeSet;
use symthaea_fabrication_kernel::attestation::SignatureAlgorithm;
use symthaea_fabrication_kernel::audit::AuditJournal;
use symthaea_fabrication_kernel::crypto_digest::{Sha256Digest, sha256};
use symthaea_fabrication_kernel::execution_guard::{
    ContainmentAction, ExecutionGuard, ExecutionGuardPolicy,
};
use symthaea_fabrication_kernel::gateway_consensus_tracker::GatewayConsensusTracker;
use symthaea_fabrication_kernel::gateway_replay::{
    build_gateway_replay_contract, verify_gateway_replay_contract,
};
use symthaea_fabrication_kernel::gateway_state::{FabricationGatewayState, GatewayStateEnvelope};
use symthaea_fabrication_kernel::incident_ledger::IncidentLedger;
use symthaea_fabrication_kernel::operator_command_tracker::OperatorCommandTracker;
use symthaea_fabrication_kernel::reconciliation::{
    append_submission_event_audit, reconcile_submission_audit,
};
use symthaea_fabrication_kernel::rotation::{
    KeyRotationPolicy, TRUST_ROTATION_SCHEMA, TrustRotationProposal, TrustRotationSigner,
    TrustRotationVerifier, authorize_trust_rotation, sign_trust_rotation_proposal,
};
use symthaea_fabrication_kernel::session::MachineSessionTracker;
use symthaea_fabrication_kernel::submission_ledger::{SubmissionIntent, SubmissionLedger};
use symthaea_fabrication_kernel::telemetry::{
    MACHINE_TELEMETRY_SCHEMA, MachineTelemetryPayload, MachineTelemetryPolicy,
    MachineTelemetrySigner, MachineTelemetryVerifier, TelemetryExpectation, sign_machine_telemetry,
    verify_machine_telemetry,
};
use symthaea_fabrication_kernel::telemetry_tracker::MachineTelemetryTracker;
use symthaea_fabrication_kernel::trust::{
    KeyLifecycleStatus, KeyTrustRecord, KeyUsage, TrustSnapshot, digest_trust_snapshot,
};

#[derive(Clone)]
struct Provider {
    algorithm: SignatureAlgorithm,
    key_id: String,
}

impl Provider {
    fn signature(&self, message: &[u8]) -> Vec<u8> {
        let mut bytes = self.key_id.as_bytes().to_vec();
        bytes.extend_from_slice(message);
        sha256(&bytes).0.to_vec()
    }
}

impl MachineTelemetrySigner for Provider {
    fn algorithm(&self) -> SignatureAlgorithm {
        self.algorithm.clone()
    }
    fn key_id(&self) -> &str {
        &self.key_id
    }
    fn sign_telemetry(&self, message: &[u8]) -> Result<Vec<u8>, String> {
        Ok(self.signature(message))
    }
}

impl MachineTelemetryVerifier for Provider {
    fn verify_telemetry(
        &self,
        algorithm: &SignatureAlgorithm,
        key_id: &str,
        message: &[u8],
        signature: &[u8],
    ) -> Result<bool, String> {
        Ok(algorithm == &self.algorithm
            && key_id == self.key_id
            && signature == self.signature(message).as_slice())
    }
}

impl TrustRotationSigner for Provider {
    fn algorithm(&self) -> SignatureAlgorithm {
        self.algorithm.clone()
    }
    fn key_id(&self) -> &str {
        &self.key_id
    }
    fn sign_rotation(&self, message: &[u8]) -> Result<Vec<u8>, String> {
        Ok(self.signature(message))
    }
}

struct RotationVerifier;

impl TrustRotationVerifier for RotationVerifier {
    fn verify_rotation(
        &self,
        _algorithm: &SignatureAlgorithm,
        key_id: &str,
        message: &[u8],
        signature: &[u8],
    ) -> Result<bool, String> {
        let mut bytes = key_id.as_bytes().to_vec();
        bytes.extend_from_slice(message);
        Ok(signature == sha256(&bytes).0.as_slice())
    }
}

fn all_usages() -> BTreeSet<KeyUsage> {
    BTreeSet::from([
        KeyUsage::FabricationManifest,
        KeyUsage::MachineSession,
        KeyUsage::MachineTelemetry,
        KeyUsage::OperatorCommand,
        KeyUsage::GatewayConsensus,
        KeyUsage::IncidentEvidence,
        KeyUsage::ReleaseCertification,
        KeyUsage::TrustRotation,
        KeyUsage::RecoveryAuthorization,
        KeyUsage::AuditAnchor,
    ])
}

fn key(algorithm: SignatureAlgorithm, key_id: &str) -> KeyTrustRecord {
    KeyTrustRecord {
        algorithm,
        key_id: key_id.into(),
        not_before_unix_s: 100,
        not_after_unix_s: Some(10_000),
        status: KeyLifecycleStatus::Active,
        usages: all_usages(),
    }
}

fn trust_snapshot() -> TrustSnapshot {
    TrustSnapshot::new(
        1,
        100,
        5_000,
        vec![
            key(SignatureAlgorithm::Ed25519, "gateway-ed"),
            key(SignatureAlgorithm::MlDsa65, "gateway-pq"),
        ],
    )
    .unwrap()
}

#[test]
fn durable_gateway_evidence_is_cross_bound_and_rotation_ready() {
    let trust = trust_snapshot();
    let ed = Provider {
        algorithm: SignatureAlgorithm::Ed25519,
        key_id: "gateway-ed".into(),
    };
    let pq = Provider {
        algorithm: SignatureAlgorithm::MlDsa65,
        key_id: "gateway-pq".into(),
    };

    let manifest_digest = sha256(b"manifest");
    let session_digest = sha256(b"session");
    let payload = MachineTelemetryPayload {
        schema_version: MACHINE_TELEMETRY_SCHEMA.into(),
        manifest_digest,
        machine_id: "machine-1".into(),
        session_digest,
        session_sequence: 3,
        printer_job_id: "job-1".into(),
        frame_sequence: 1,
        observed_at_unix_ms: 500_000,
        elapsed_ms: 2_000,
        heartbeat_sequence: 1,
        progress_ppm: 100_000,
        nozzle_actual_milli_c: 200_000,
        nozzle_target_milli_c: 200_000,
        bed_actual_milli_c: 60_000,
        bed_target_milli_c: 60_000,
    };
    let signed = sign_machine_telemetry(payload, &ed).unwrap();
    let verified = verify_machine_telemetry(
        signed,
        &MachineTelemetryPolicy::default(),
        TelemetryExpectation {
            manifest_digest,
            machine_id: "machine-1",
            session_digest,
            session_sequence: 3,
            printer_job_id: "job-1",
        },
        &trust,
        500_100,
        &ed,
    )
    .unwrap();
    let mut telemetry_tracker = MachineTelemetryTracker::default();
    telemetry_tracker.accept(&verified).unwrap();
    let mut guard = ExecutionGuard::new(ExecutionGuardPolicy::default()).unwrap();
    let guarded = guard.observe_verified(&verified);
    assert_eq!(guarded.decision.latched_action, ContainmentAction::Continue);
    assert_eq!(guarded.telemetry_digest, verified.telemetry_digest());

    let mut ledger = SubmissionLedger::default();
    let intent = || SubmissionIntent {
        request_id: "request-1",
        manifest_digest,
        machine_id: "machine-1",
        session_digest,
        session_sequence: 3,
    };
    ledger.prepare(501_000, intent()).unwrap();
    ledger
        .mark_uncertain(502_000, intent(), sha256(b"transport timeout"))
        .unwrap();
    ledger.reconcile(503_000, intent(), "job-1").unwrap();

    let mut audit = AuditJournal::default();
    for event in &ledger.events {
        append_submission_event_audit(&mut audit, "gateway", event).unwrap();
    }
    let reconciliation = reconcile_submission_audit(&ledger, &audit).unwrap();
    assert!(reconciliation.reconciled());

    let state = FabricationGatewayState::genesis(
        600_000,
        trust.clone(),
        audit,
        MachineSessionTracker::default(),
        telemetry_tracker,
        ledger,
        OperatorCommandTracker::default(),
        GatewayConsensusTracker::default(),
        IncidentLedger::default(),
    )
    .unwrap();
    let envelope = GatewayStateEnvelope::seal(state.clone()).unwrap();
    assert_eq!(
        GatewayStateEnvelope::from_bytes(&envelope.to_bytes().unwrap())
            .unwrap()
            .open()
            .unwrap(),
        state
    );

    let operational_replay_digest = Sha256Digest([9; 32]);
    let rotation_policy = KeyRotationPolicy::default();
    let gateway_replay = build_gateway_replay_contract(
        &state,
        operational_replay_digest,
        &reconciliation,
        Some(&rotation_policy),
    )
    .unwrap();
    assert!(
        verify_gateway_replay_contract(
            &gateway_replay,
            &state,
            operational_replay_digest,
            &reconciliation,
            Some(&rotation_policy),
        )
        .unwrap()
        .reproducible()
    );

    let mut proposed_keys = trust.keys.clone();
    proposed_keys.push(key(SignatureAlgorithm::MlDsa87, "gateway-pq-next"));
    let proposal = TrustRotationProposal {
        schema_version: TRUST_ROTATION_SCHEMA.into(),
        current_snapshot_digest: digest_trust_snapshot(&trust).unwrap(),
        proposed_snapshot: TrustSnapshot::new(2, 700, 6_000, proposed_keys).unwrap(),
        activates_at_unix_s: 700,
        emergency: false,
        reason_digest: sha256(b"scheduled gateway key rotation"),
    };
    let signed_rotation = sign_trust_rotation_proposal(proposal, &[&ed, &pq]).unwrap();
    let authorized_rotation = authorize_trust_rotation(
        signed_rotation,
        &trust,
        &rotation_policy,
        650,
        &RotationVerifier,
    )
    .unwrap();
    assert_eq!(authorized_rotation.valid_signers().len(), 2);
    assert_eq!(authorized_rotation.proposal().proposed_snapshot.sequence, 2);
}
