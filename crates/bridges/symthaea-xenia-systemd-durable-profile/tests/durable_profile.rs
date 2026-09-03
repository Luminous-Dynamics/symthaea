// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::collections::{BTreeSet, VecDeque};
use std::fs;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use ed25519_dalek::{Signer, SigningKey};
use symthaea_action_checkpoint::CheckpointHead;
use symthaea_action_runtime::{ExecutionId, ReservationId};
use symthaea_authority::{AuthorityEpoch, CapabilityGrant, Digest32, PrincipalId, TaskId};
use symthaea_authority_frontier_sqlite::SqliteCheckpointCasStore;
use symthaea_authority_time::{
    AUTHORITY_TIME_SCHEMA_VERSION, AuthorityTimeStatementV1, PendingAuthorityTimeChallenge,
    TimeAuthorityId, TrustedTimeAuthorityV1, TrustedTimePolicyV1, VerifiedAuthorityTime,
    verify_authority_time_v1,
};
use symthaea_system_attempt_evidence::{
    AttemptEvidenceJournal, AttemptEvidenceState, SqliteAttemptEvidenceJournal,
};
use symthaea_system_broker::{
    DispatchEvidence, HostId, RestartPlan, ServiceBackend, ServiceObservation, ServiceUnit,
    VerificationResult, restart_risk_charge,
};
use symthaea_xenia_authority::{
    AGENT_CAPABILITY_ATTESTATION_SCHEMA, ED25519_SIGNATURE_ALGORITHM, ExecutorWorkloadV1,
    TranscriptSignatureSuiteV1, XENIA_LEDGER_CHECKPOINT_SCHEMA, XeniaAgentAuthorizationV1,
    XeniaAgentCapabilityAttestationV1, XeniaCheckpointAnchorV1, XeniaFreshnessPolicyV1,
    XeniaLedgerCheckpointV1, XeniaSessionExpectationV1, XeniaSignatureEnvelopeV1,
    verify_xenia_capability_v1,
};
use symthaea_xenia_systemd_durable_profile::{
    DurableAttemptEvidenceStatus, DurableRecoveryError, DurableXeniaSystemdBootstrap,
};

static NEXT_DB: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Copy)]
struct FakeBackendError;
impl std::fmt::Display for FakeBackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("fake backend error")
    }
}
impl std::error::Error for FakeBackendError {}

struct FakeBackend {
    observations: VecDeque<ServiceObservation>,
    restart_calls: Arc<Mutex<usize>>,
}

impl FakeBackend {
    fn new(observations: Vec<ServiceObservation>, restart_calls: Arc<Mutex<usize>>) -> Self {
        Self {
            observations: observations.into(),
            restart_calls,
        }
    }
}

impl ServiceBackend for FakeBackend {
    type Error = FakeBackendError;

    fn observe(
        &mut self,
        _host: &HostId,
        _unit: &ServiceUnit,
    ) -> Result<ServiceObservation, Self::Error> {
        self.observations.pop_front().ok_or(FakeBackendError)
    }

    fn restart(
        &mut self,
        _host: &HostId,
        _unit: &ServiceUnit,
    ) -> Result<DispatchEvidence, Self::Error> {
        let mut calls = self.restart_calls.lock().map_err(|_| FakeBackendError)?;
        *calls += 1;
        Ok(DispatchEvidence::Applied)
    }
}

fn db_path() -> std::path::PathBuf {
    let id = NEXT_DB.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!(
        "symthaea-xenia-systemd-durable-{}-{id}.sqlite",
        std::process::id()
    ))
}

fn cleanup(path: &std::path::Path) {
    let _ = fs::remove_file(path);
    let _ = fs::remove_file(format!("{}-wal", path.display()));
    let _ = fs::remove_file(format!("{}-shm", path.display()));
}

fn host() -> HostId {
    HostId::parse("host-a").unwrap()
}

fn unit() -> ServiceUnit {
    ServiceUnit::parse("postgresql.service").unwrap()
}

fn observation(active: &str, sub: &str, invocation: &str) -> ServiceObservation {
    ServiceObservation {
        host: host(),
        unit: unit(),
        active_state: active.into(),
        sub_state: sub.into(),
        invocation_id: Some(invocation.into()),
    }
}

fn plan_grant_workload() -> (ServiceObservation, RestartPlan, CapabilityGrant, ExecutorWorkloadV1) {
    let before = observation("failed", "failed", "invocation-old");
    let actor = PrincipalId("symthaea://agent/system-recovery".into());
    let executor = PrincipalId("spiffe://luminous.local/symthaea/system-broker".into());
    let task = Some(TaskId("repair-postgresql".into()));
    let plan = RestartPlan::new(actor.clone(), executor.clone(), task.clone(), &before);

    let mut grant = CapabilityGrant::new(
        "xenia-systemd-durable-grant-1",
        PrincipalId("xenia://operator/alice".into()),
        actor,
        AuthorityEpoch(7),
    );
    grant.audience = Some(executor.clone());
    grant.task = task;
    grant.resources = BTreeSet::from([plan.resource()]);
    grant.operations = BTreeSet::from([plan.operation()]);
    grant.plan_digest = Some(plan.digest());
    grant.world_digest = Some(plan.world_digest);
    grant.expires_at_unix_s = Some(180);
    grant.max_uses = 1;
    grant.risk_budget = restart_risk_charge();

    let workload = ExecutorWorkloadV1 {
        executor,
        artifact_digest: Digest32([31; 32]),
        configuration_digest: Digest32([32; 32]),
        host_identity_digest: Digest32([33; 32]),
    };
    (before, plan, grant, workload)
}

fn verified_time(grant: &CapabilityGrant, witnessed_unix_s: u64) -> VerifiedAuthorityTime {
    let key_a = SigningKey::from_bytes(&[61; 32]);
    let key_b = SigningKey::from_bytes(&[62; 32]);
    let policy = TrustedTimePolicyV1 {
        schema_version: AUTHORITY_TIME_SCHEMA_VERSION,
        policy_id: [63; 16],
        authorities: vec![
            TrustedTimeAuthorityV1 {
                authority_id: TimeAuthorityId([1; 16]),
                verifying_key: key_a.verifying_key().to_bytes(),
                organization_binding: [71; 32],
                service_binding: [81; 32],
            },
            TrustedTimeAuthorityV1 {
                authority_id: TimeAuthorityId([2; 16]),
                verifying_key: key_b.verifying_key().to_bytes(),
                organization_binding: [72; 32],
                service_binding: [82; 32],
            },
        ],
        threshold: 2,
        minimum_organizations: 2,
        maximum_uncertainty_s: 1,
        maximum_challenge_age_ns: 5_000_000_000,
        maximum_post_verification_age_ns: 5_000_000_000,
    };
    let pending = PendingAuthorityTimeChallenge::new(&policy, grant.digest().0).unwrap();
    let challenge = pending.wire();
    let sign = |authority_id: TimeAuthorityId, key: &SigningKey| {
        let mut statement = AuthorityTimeStatementV1 {
            schema_version: AUTHORITY_TIME_SCHEMA_VERSION,
            authority_id,
            policy_digest: challenge.policy_digest,
            subject_digest: challenge.subject_digest,
            challenge_nonce: challenge.nonce,
            witnessed_unix_s,
            uncertainty_s: 1,
            signature: [0; 64],
        };
        statement.signature = key.sign(&statement.canonical_message().unwrap()).to_bytes();
        statement
    };
    verify_authority_time_v1(
        &policy,
        pending,
        &[
            sign(TimeAuthorityId([1; 16]), &key_a),
            sign(TimeAuthorityId([2; 16]), &key_b),
        ],
    )
    .unwrap()
}

fn verified_proof(
    grant: &CapabilityGrant,
    workload: &ExecutorWorkloadV1,
    agent_head: CheckpointHead,
    authority_time: &VerifiedAuthorityTime,
) -> symthaea_xenia_authority::VerifiedXeniaCapability {
    let signing_key = SigningKey::from_bytes(&[3; 32]);
    let public_key = signing_key.verifying_key().to_bytes();
    let authorization = XeniaAgentAuthorizationV1 {
        schema_version: 1,
        authorization_id: [41; 16],
        session_id: [42; 16],
        session_transcript_hash: [43; 32],
        session_signature_suite: TranscriptSignatureSuiteV1::Ed25519Rfc8032,
        capability_digest: grant.digest().0,
        executor_workload_digest: workload.digest().unwrap().0,
        authority_epoch: grant.authority_epoch.0,
        issued_at_unix_s: 100,
        expires_at_unix_s: 160,
        nonce: [44; 16],
        ledger_entry_count: 20,
        ledger_head_hash: [45; 32],
        prior_checkpoint: Some(XeniaCheckpointAnchorV1 {
            sequence: agent_head.sequence,
            digest: agent_head.digest.0,
        }),
    };
    let attestation = XeniaAgentCapabilityAttestationV1 {
        schema: AGENT_CAPABILITY_ATTESTATION_SCHEMA.into(),
        ledger_public_key_fingerprint: *blake3::hash(&public_key).as_bytes(),
        signature: XeniaSignatureEnvelopeV1 {
            algorithm: ED25519_SIGNATURE_ALGORITHM.into(),
            signature: signing_key
                .sign(&authorization.canonical_message().unwrap())
                .to_bytes()
                .to_vec(),
        },
        authorization,
    };

    let mut checkpoint = XeniaLedgerCheckpointV1 {
        schema: XENIA_LEDGER_CHECKPOINT_SCHEMA.into(),
        entry_count: 20,
        head_hash: [45; 32],
        ledger_public_key: public_key,
        timestamp_unix_secs: 120,
        signature: Vec::new(),
    };
    checkpoint.signature = signing_key
        .sign(&checkpoint.signature_message().unwrap())
        .to_bytes()
        .to_vec();

    verify_xenia_capability_v1(
        &attestation,
        &checkpoint,
        public_key,
        grant,
        workload,
        XeniaSessionExpectationV1 {
            session_id: [42; 16],
            transcript_hash: [43; 32],
            transcript_signature_suite: TranscriptSignatureSuiteV1::Ed25519Rfc8032,
        },
        agent_head,
        authority_time,
        XeniaFreshnessPolicyV1::strict(30, 5),
    )
    .unwrap()
}

#[test]
fn durable_profile_reopens_exact_xenia_attempt_and_checkpoint_lineage() {
    let path = db_path();
    cleanup(&path);
    let (before, plan, grant, workload) = plan_grant_workload();
    let calls = Arc::new(Mutex::new(0usize));
    let backend = FakeBackend::new(
        vec![
            before.clone(),
            before,
            observation("active", "running", "invocation-new"),
        ],
        calls.clone(),
    );

    let bootstrap = DurableXeniaSystemdBootstrap::bootstrap(grant.clone(), backend, &path).unwrap();
    let authorized_head = bootstrap.authorization_checkpoint_head();
    let time = verified_time(&grant, 125);
    let proof = verified_proof(&grant, &workload, authorized_head, &time);
    let receipt = bootstrap
        .recover_verified_once(
            proof,
            &time,
            AuthorityEpoch(7),
            &plan,
            ExecutionId("exec-durable-1".into()),
            ReservationId("reservation-durable-1".into()),
            &[],
        )
        .unwrap();

    assert_eq!(*calls.lock().unwrap(), 1);
    assert_eq!(receipt.recovery.verification, VerificationResult::Healthy);
    assert_eq!(receipt.recovery.use_state.committed, 1);
    assert!(matches!(
        receipt.attempt_evidence,
        DurableAttemptEvidenceStatus::RecoveryCompleted(head) if head.sequence == 2
    ));

    let journal = SqliteAttemptEvidenceJournal::open(&path).unwrap();
    let chain = journal.load_chain(receipt.attempt_key).unwrap();
    assert_eq!(chain.len(), 3);
    assert_eq!(chain[0].state, AttemptEvidenceState::DispatchArmed);
    assert_eq!(chain[1].state, AttemptEvidenceState::Applied);
    assert_eq!(chain[2].state, AttemptEvidenceState::RecoveryCompleted);
    assert_eq!(
        chain[0].context.authority_evidence_digest,
        Some(receipt.authority_evidence_digest)
    );
    assert_eq!(chain[2].recovery_outcome, Some(receipt.recovery.outcome));
    assert_eq!(chain[2].verification, Some(VerificationResult::Healthy));

    let checkpoint_store = SqliteCheckpointCasStore::open(&path).unwrap();
    let (_, durable_head) = checkpoint_store.load_frontier().unwrap().unwrap();
    assert_eq!(durable_head, receipt.recovery.checkpoint_head);
    assert_ne!(durable_head, authorized_head);
    cleanup(&path);
}

#[test]
fn newer_verified_time_rejects_expired_xenia_proof_before_effect_or_attempt_evidence() {
    let path = db_path();
    cleanup(&path);
    let (before, plan, grant, workload) = plan_grant_workload();
    let calls = Arc::new(Mutex::new(0usize));
    let backend = FakeBackend::new(vec![before], calls.clone());
    let bootstrap = DurableXeniaSystemdBootstrap::bootstrap(grant.clone(), backend, &path).unwrap();
    let head = bootstrap.authorization_checkpoint_head();
    let verification_time = verified_time(&grant, 125);
    let proof = verified_proof(&grant, &workload, head, &verification_time);
    let effect_entry_time = verified_time(&grant, 161);

    assert!(matches!(
        bootstrap.recover_verified_once(
            proof,
            &effect_entry_time,
            AuthorityEpoch(7),
            &plan,
            ExecutionId("exec-expired".into()),
            ReservationId("reservation-expired".into()),
            &[],
        ),
        Err(DurableRecoveryError::XeniaProofExpiredAtEffectEntry)
    ));
    assert_eq!(*calls.lock().unwrap(), 0);

    let journal = SqliteAttemptEvidenceJournal::open(&path).unwrap();
    // No attempt key was ever materialized into the journal because failure
    // happened before effect admission/instrumentation.
    let count: i64 = rusqlite::Connection::open(&path)
        .unwrap()
        .query_row("SELECT COUNT(*) FROM system_attempt_evidence", [], |row| row.get(0))
        .unwrap();
    assert_eq!(count, 0);
    drop(journal);
    cleanup(&path);
}
