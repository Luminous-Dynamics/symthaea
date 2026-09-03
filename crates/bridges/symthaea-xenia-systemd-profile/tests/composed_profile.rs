// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::collections::{BTreeSet, VecDeque};
use std::fmt;
use std::sync::{Arc, Mutex};

use ed25519_dalek::{Signer, SigningKey};
use symthaea_action_checkpoint::{CheckpointHead, GrantAccountCheckpoint};
use symthaea_action_runtime::{ExecutionId, GrantUseState, ReservationId};
use symthaea_authority::{
    AuthorityContext, AuthorityEpoch, CapabilityGrant, Digest32, PrincipalId, TaskId,
};
use symthaea_authority_frontier::CheckpointCasStore;
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
use symthaea_xenia_systemd_profile::{
    ProfileRecoveryError, XeniaSystemdRecoveryProfile,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CasConflict;
impl fmt::Display for CasConflict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str("CAS conflict")
    }
}
impl std::error::Error for CasConflict {}

#[derive(Clone, Default)]
struct SharedCasStore {
    head: Arc<Mutex<Option<CheckpointHead>>>,
}

impl CheckpointCasStore for SharedCasStore {
    type Error = CasConflict;

    fn compare_and_swap(
        &mut self,
        expected_previous: Option<CheckpointHead>,
        checkpoint: &GrantAccountCheckpoint,
    ) -> Result<CheckpointHead, Self::Error> {
        let mut head = self.head.lock().map_err(|_| CasConflict)?;
        if *head != expected_previous {
            return Err(CasConflict);
        }
        let next = checkpoint.head().map_err(|_| CasConflict)?;
        *head = Some(next);
        Ok(next)
    }
}

#[derive(Debug, Clone, Copy)]
struct FakeBackendError;
impl fmt::Display for FakeBackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
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

fn plan_and_grant() -> (ServiceObservation, RestartPlan, CapabilityGrant, ExecutorWorkloadV1) {
    let before = observation("failed", "failed", "invocation-old");
    let actor = PrincipalId("symthaea://agent/system-recovery".into());
    let executor = PrincipalId("spiffe://luminous.local/symthaea/system-broker".into());
    let task = Some(TaskId("repair-postgresql".into()));
    let plan = RestartPlan::new(actor.clone(), executor.clone(), task.clone(), &before);

    let mut grant = CapabilityGrant::new(
        "xenia-systemd-grant-1",
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

fn verified_proof(
    grant: &CapabilityGrant,
    workload: &ExecutorWorkloadV1,
    agent_head: CheckpointHead,
    now: u64,
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
        now,
        XeniaFreshnessPolicyV1::strict(30, 5),
    )
    .unwrap()
}

fn context(now: u64) -> AuthorityContext {
    AuthorityContext {
        now_unix_s: now,
        current_epoch: AuthorityEpoch(7),
        // #305 intentionally ignores caller-provided accounting and uses its
        // actual GrantAccount state.
        use_state: GrantUseState::default(),
    }
}

#[test]
fn exact_xenia_proof_drives_one_cas_backed_typed_restart() {
    let (before, plan, grant, workload) = plan_and_grant();
    let calls = Arc::new(Mutex::new(0usize));
    let backend = FakeBackend::new(
        vec![
            before.clone(),
            before.clone(),
            observation("active", "running", "invocation-new"),
        ],
        calls.clone(),
    );
    let (mut profile, frontier) =
        XeniaSystemdRecoveryProfile::bootstrap(grant.clone(), backend, SharedCasStore::default())
            .unwrap();
    assert_eq!(profile.authorization_checkpoint_head().unwrap(), frontier.head);

    let proof = verified_proof(&grant, &workload, frontier.head, 125);
    let receipt = profile
        .recover_verified_once(
            proof,
            &plan,
            ExecutionId("exec-1".into()),
            ReservationId("reservation-1".into()),
            context(125),
            &[],
        )
        .unwrap();

    assert_eq!(*calls.lock().unwrap(), 1);
    assert_eq!(receipt.recovery.verification, VerificationResult::Healthy);
    assert_ne!(receipt.recovery.checkpoint_head, frontier.head);
    assert_eq!(receipt.recovery.use_state.committed, 1);
}

#[test]
fn two_processes_can_verify_same_attestation_but_only_one_crosses_cas() {
    let (before, plan, grant, workload) = plan_and_grant();
    let shared_store = SharedCasStore::default();

    let calls_a = Arc::new(Mutex::new(0usize));
    let backend_a = FakeBackend::new(
        vec![
            before.clone(),
            before.clone(),
            observation("active", "running", "invocation-a"),
        ],
        calls_a.clone(),
    );
    let (mut profile_a, frontier) =
        XeniaSystemdRecoveryProfile::bootstrap(grant.clone(), backend_a, shared_store.clone())
            .unwrap();

    let calls_b = Arc::new(Mutex::new(0usize));
    let backend_b = FakeBackend::new(vec![before.clone()], calls_b.clone());
    let mut profile_b = XeniaSystemdRecoveryProfile::restore(
        grant.clone(),
        frontier.checkpoint.clone(),
        frontier.head,
        backend_b,
        shared_store,
    )
    .unwrap();

    // Cryptographic verification is intentionally replayable by itself. The
    // durable CAS frontier is what prevents two processes from both entering
    // the effect lineage.
    let proof_a = verified_proof(&grant, &workload, frontier.head, 125);
    let proof_b = verified_proof(&grant, &workload, frontier.head, 125);

    profile_a
        .recover_verified_once(
            proof_a,
            &plan,
            ExecutionId("exec-a".into()),
            ReservationId("reservation-a".into()),
            context(125),
            &[],
        )
        .unwrap();

    let second = profile_b.recover_verified_once(
        proof_b,
        &plan,
        ExecutionId("exec-b".into()),
        ReservationId("reservation-b".into()),
        context(125),
        &[],
    );
    assert!(matches!(second, Err(ProfileRecoveryError::Broker(_))));
    assert_eq!(*calls_a.lock().unwrap(), 1);
    assert_eq!(*calls_b.lock().unwrap(), 0);
    assert!(profile_b.is_contained());
}

#[test]
fn proof_is_rechecked_for_expiry_at_effect_entry() {
    let (before, plan, grant, workload) = plan_and_grant();
    let calls = Arc::new(Mutex::new(0usize));
    let backend = FakeBackend::new(vec![before], calls.clone());
    let (mut profile, frontier) =
        XeniaSystemdRecoveryProfile::bootstrap(grant.clone(), backend, SharedCasStore::default())
            .unwrap();
    let proof = verified_proof(&grant, &workload, frontier.head, 125);

    assert!(matches!(
        profile.recover_verified_once(
            proof,
            &plan,
            ExecutionId("exec-expired".into()),
            ReservationId("reservation-expired".into()),
            context(161),
            &[],
        ),
        Err(ProfileRecoveryError::XeniaProofExpiredAtEffectEntry)
    ));
    assert_eq!(*calls.lock().unwrap(), 0);
    assert_eq!(profile.authorization_checkpoint_head().unwrap(), frontier.head);
}
