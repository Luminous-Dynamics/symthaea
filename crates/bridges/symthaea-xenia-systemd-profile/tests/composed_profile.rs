// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use std::collections::{BTreeSet, VecDeque};
use std::fmt;
use std::sync::{Arc, Mutex};

use ed25519_dalek::{Signer, SigningKey};
use symthaea_action_checkpoint::{CheckpointHead, GrantAccountCheckpoint};
use symthaea_action_runtime::{ExecutionId, ReservationId};
use symthaea_authority::{
    AuthorityEpoch, CapabilityGrant, Digest32, NegativeAuthorityFact, PrincipalId, TaskId,
};
use symthaea_authority_frontier::CheckpointCasStore;
use symthaea_authority_state::{
    AUTHORITY_STATE_SCHEMA_VERSION, AuthorityStatePolicyV1, AuthorityStateStatementV1,
    AuthorityStateWitnessId, PendingAuthorityStateChallenge, TrustedAuthorityStateWitnessV1,
    VerifiedAuthorityState, verify_authority_state_v1,
};
use symthaea_authority_time::{
    AUTHORITY_TIME_SCHEMA_VERSION, AuthorityTimeStatementV1, PendingAuthorityTimeChallenge,
    TimeAuthorityId, TrustedTimeAuthorityV1, TrustedTimePolicyV1, VerifiedAuthorityTime,
    verify_authority_time_v1,
};
use symthaea_executor_workload::{
    EXECUTOR_WORKLOAD_SCHEMA_VERSION, ExecutorWorkloadV1, PendingWorkloadChallenge,
    TrustedWorkloadWitnessV1, VerifiedExecutorWorkload, WorkloadMeasurementStatementV1,
    WorkloadWitnessId, WorkloadWitnessPolicyV1, measure_linux_process_instance,
    verify_executor_workload_v1,
};
use symthaea_system_broker::{
    DispatchEvidence, HostId, RestartPlan, ServiceBackend, ServiceObservation, ServiceUnit,
    VerificationResult, restart_risk_charge,
};
use symthaea_xenia_authority::{
    AGENT_CAPABILITY_ATTESTATION_SCHEMA, ED25519_SIGNATURE_ALGORITHM,
    TranscriptSignatureSuiteV1, XENIA_LEDGER_CHECKPOINT_SCHEMA, XeniaAgentAuthorizationV1,
    XeniaAgentCapabilityAttestationV1, XeniaCheckpointAnchorV1, XeniaFreshnessPolicyV1,
    XeniaLedgerCheckpointV1, XeniaSessionExpectationV1, XeniaSignatureEnvelopeV1,
    verify_xenia_capability_v1,
};
use symthaea_xenia_systemd_profile::{ProfileRecoveryError, XeniaSystemdRecoveryProfile};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CasConflict;
impl fmt::Display for CasConflict {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.write_str("CAS conflict") }
}
impl std::error::Error for CasConflict {}

#[derive(Clone, Default)]
struct SharedCasStore { head: Arc<Mutex<Option<CheckpointHead>>> }
impl CheckpointCasStore for SharedCasStore {
    type Error = CasConflict;
    fn compare_and_swap(
        &mut self,
        expected_previous: Option<CheckpointHead>,
        checkpoint: &GrantAccountCheckpoint,
    ) -> Result<CheckpointHead, Self::Error> {
        let mut head = self.head.lock().map_err(|_| CasConflict)?;
        if *head != expected_previous { return Err(CasConflict); }
        let next = checkpoint.head().map_err(|_| CasConflict)?;
        *head = Some(next);
        Ok(next)
    }
}

#[derive(Debug, Clone, Copy)]
struct FakeBackendError;
impl fmt::Display for FakeBackendError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { f.write_str("fake backend") }
}
impl std::error::Error for FakeBackendError {}

struct FakeBackend {
    observations: VecDeque<ServiceObservation>,
    restart_calls: Arc<Mutex<usize>>,
}
impl ServiceBackend for FakeBackend {
    type Error = FakeBackendError;
    fn observe(&mut self, _host: &HostId, _unit: &ServiceUnit) -> Result<ServiceObservation, Self::Error> {
        self.observations.pop_front().ok_or(FakeBackendError)
    }
    fn restart(&mut self, _host: &HostId, _unit: &ServiceUnit) -> Result<DispatchEvidence, Self::Error> {
        *self.restart_calls.lock().map_err(|_| FakeBackendError)? += 1;
        Ok(DispatchEvidence::Applied)
    }
}

fn host() -> HostId { HostId::parse("host-a").unwrap() }
fn unit() -> ServiceUnit { ServiceUnit::parse("postgresql.service").unwrap() }
fn observation(active: &str, sub: &str, invocation: &str) -> ServiceObservation {
    ServiceObservation {
        host: host(),
        unit: unit(),
        active_state: active.into(),
        sub_state: sub.into(),
        invocation_id: Some(invocation.into()),
    }
}

fn plan_and_grant() -> (ServiceObservation, RestartPlan, CapabilityGrant) {
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
    grant.audience = Some(executor);
    grant.task = task;
    grant.resources = BTreeSet::from([plan.resource()]);
    grant.operations = BTreeSet::from([plan.operation()]);
    grant.plan_digest = Some(plan.digest());
    grant.world_digest = Some(plan.world_digest);
    grant.expires_at_unix_s = Some(180);
    grant.max_uses = 1;
    grant.risk_budget = restart_risk_charge();
    (before, plan, grant)
}

fn verified_time(grant: &CapabilityGrant, witnessed: u64) -> VerifiedAuthorityTime {
    let a = SigningKey::from_bytes(&[31; 32]);
    let b = SigningKey::from_bytes(&[32; 32]);
    let policy = TrustedTimePolicyV1 {
        schema_version: AUTHORITY_TIME_SCHEMA_VERSION,
        policy_id: [33; 16],
        authorities: vec![
            TrustedTimeAuthorityV1 { authority_id: TimeAuthorityId([1;16]), verifying_key: a.verifying_key().to_bytes(), organization_binding:[41;32], service_binding:[51;32] },
            TrustedTimeAuthorityV1 { authority_id: TimeAuthorityId([2;16]), verifying_key: b.verifying_key().to_bytes(), organization_binding:[42;32], service_binding:[52;32] },
        ],
        threshold: 2,
        minimum_organizations: 2,
        maximum_uncertainty_s: 1,
        maximum_challenge_age_ns: 5_000_000_000,
        maximum_post_verification_age_ns: 5_000_000_000,
    };
    let pending = PendingAuthorityTimeChallenge::new(&policy, grant.digest().0).unwrap();
    let c = pending.wire();
    let sign = |id: TimeAuthorityId, key: &SigningKey| {
        let mut s = AuthorityTimeStatementV1 { schema_version:1, authority_id:id, policy_digest:c.policy_digest, subject_digest:c.subject_digest, challenge_nonce:c.nonce, witnessed_unix_s:witnessed, uncertainty_s:1, signature:Vec::new() };
        s.signature = key.sign(&s.canonical_message().unwrap()).to_bytes().to_vec(); s
    };
    verify_authority_time_v1(&policy, pending, &[sign(TimeAuthorityId([1;16]), &a), sign(TimeAuthorityId([2;16]), &b)]).unwrap()
}

fn verified_state(grant: &CapabilityGrant, time: &VerifiedAuthorityTime, facts: Vec<NegativeAuthorityFact>) -> VerifiedAuthorityState {
    let a = SigningKey::from_bytes(&[61; 32]);
    let b = SigningKey::from_bytes(&[62; 32]);
    let policy = AuthorityStatePolicyV1 {
        schema_version: AUTHORITY_STATE_SCHEMA_VERSION,
        policy_id: [63;16],
        witnesses: vec![
            TrustedAuthorityStateWitnessV1 { witness_id:AuthorityStateWitnessId([1;16]), verifying_key:a.verifying_key().to_bytes(), organization_binding:[71;32], service_binding:[81;32] },
            TrustedAuthorityStateWitnessV1 { witness_id:AuthorityStateWitnessId([2;16]), verifying_key:b.verifying_key().to_bytes(), organization_binding:[72;32], service_binding:[82;32] },
        ],
        threshold:2,
        minimum_organizations:2,
        maximum_challenge_age_s:60,
        maximum_post_verification_age_s:60,
    };
    let pending = PendingAuthorityStateChallenge::new(&policy, grant, time).unwrap();
    let c = pending.wire();
    let sign = |id:AuthorityStateWitnessId,key:&SigningKey,generation:u64| {
        let mut s = AuthorityStateStatementV1 { schema_version:1,witness_id:id,challenge_nonce:c.nonce,grant_digest:c.grant_digest,state_policy_digest:c.state_policy_digest,time_policy_digest:c.time_policy_digest,source_frontier_sequence:20,source_frontier_digest:Digest32([45;32]),state_sequence:20,authority_epoch:grant.authority_epoch,negative_facts:facts.clone(),witness_generation:generation,signature:Vec::new() };
        s.signature=key.sign(&s.canonical_message().unwrap()).to_bytes().to_vec(); s
    };
    verify_authority_state_v1(&policy,grant,pending,time,&[sign(AuthorityStateWitnessId([1;16]),&a,1),sign(AuthorityStateWitnessId([2;16]),&b,2)]).unwrap()
}

fn verified_workload(grant:&CapabilityGrant,time:&VerifiedAuthorityTime,start_tick_delta:u64) -> (VerifiedExecutorWorkload, ExecutorWorkloadV1) {
    let direct = measure_linux_process_instance(std::process::id()).unwrap();
    let mut process = direct.process;
    process.start_time_ticks = process.start_time_ticks.saturating_add(start_tick_delta);
    let workload = ExecutorWorkloadV1 { executor:grant.audience.clone().unwrap(), artifact_digest:direct.artifact_digest, configuration_digest:Digest32([91;32]), host_identity_digest:direct.host_identity_digest };
    let a=SigningKey::from_bytes(&[101;32]); let b=SigningKey::from_bytes(&[102;32]);
    let policy=WorkloadWitnessPolicyV1 { schema_version:EXECUTOR_WORKLOAD_SCHEMA_VERSION,policy_id:[103;16],witnesses:vec![
        TrustedWorkloadWitnessV1{witness_id:WorkloadWitnessId([1;16]),verifying_key:a.verifying_key().to_bytes(),organization_binding:[111;32],service_binding:[121;32]},
        TrustedWorkloadWitnessV1{witness_id:WorkloadWitnessId([2;16]),verifying_key:b.verifying_key().to_bytes(),organization_binding:[112;32],service_binding:[122;32]},
    ],threshold:2,minimum_organizations:2,maximum_challenge_age_s:10,maximum_post_verification_age_s:10,require_nix_store_executable:false};
    let pending=PendingWorkloadChallenge::new(&policy,grant,time).unwrap(); let c=pending.wire();
    let sign=|id:WorkloadWitnessId,key:&SigningKey,generation:u64| { let mut s=WorkloadMeasurementStatementV1{schema_version:1,witness_id:id,challenge_nonce:c.nonce,grant_digest:c.grant_digest,executor:c.executor.clone(),workload_policy_digest:c.workload_policy_digest,time_policy_digest:c.time_policy_digest,workload:workload.clone(),process,witness_generation:generation,executable_in_nix_store:direct.executable_in_nix_store,signature:Vec::new()}; s.signature=key.sign(&s.canonical_message().unwrap()).to_bytes().to_vec(); s };
    let verified=verify_executor_workload_v1(&policy,grant,pending,time,&[sign(WorkloadWitnessId([1;16]),&a,1),sign(WorkloadWitnessId([2;16]),&b,2)]).unwrap();
    (verified,workload)
}

fn verified_xenia(grant:&CapabilityGrant, time:&VerifiedAuthorityTime, state:VerifiedAuthorityState, workload_proof:VerifiedExecutorWorkload, raw_workload:&ExecutorWorkloadV1, head:CheckpointHead) -> symthaea_xenia_authority::VerifiedXeniaCapability {
    let key=SigningKey::from_bytes(&[3;32]); let public_key=key.verifying_key().to_bytes();
    let authorization=XeniaAgentAuthorizationV1{schema_version:1,authorization_id:[41;16],session_id:[42;16],session_transcript_hash:[43;32],session_signature_suite:TranscriptSignatureSuiteV1::Ed25519Rfc8032,capability_digest:grant.digest().0,executor_workload_digest:raw_workload.digest().unwrap().0,authority_epoch:grant.authority_epoch.0,issued_at_unix_s:100,expires_at_unix_s:160,nonce:[44;16],ledger_entry_count:20,ledger_head_hash:[45;32],prior_checkpoint:Some(XeniaCheckpointAnchorV1{sequence:head.sequence,digest:head.digest.0})};
    let attestation=XeniaAgentCapabilityAttestationV1{schema:AGENT_CAPABILITY_ATTESTATION_SCHEMA.into(),ledger_public_key_fingerprint:*blake3::hash(&public_key).as_bytes(),signature:XeniaSignatureEnvelopeV1{algorithm:ED25519_SIGNATURE_ALGORITHM.into(),signature:key.sign(&authorization.canonical_message().unwrap()).to_bytes().to_vec()},authorization};
    let mut checkpoint=XeniaLedgerCheckpointV1{schema:XENIA_LEDGER_CHECKPOINT_SCHEMA.into(),entry_count:20,head_hash:[45;32],ledger_public_key:public_key,timestamp_unix_secs:120,signature:Vec::new()}; checkpoint.signature=key.sign(&checkpoint.signature_message().unwrap()).to_bytes().to_vec();
    verify_xenia_capability_v1(&attestation,&checkpoint,public_key,grant,workload_proof,XeniaSessionExpectationV1{session_id:[42;16],transcript_hash:[43;32],transcript_signature_suite:TranscriptSignatureSuiteV1::Ed25519Rfc8032},head,time,state,XeniaFreshnessPolicyV1::strict(30,5)).unwrap()
}

#[test]
fn measured_current_process_drives_one_typed_restart() {
    let (before,plan,grant)=plan_and_grant(); let calls=Arc::new(Mutex::new(0));
    let backend=FakeBackend{observations:vec![before.clone(),before,observation("active","running","invocation-new")].into(),restart_calls:calls.clone()};
    let (mut profile,frontier)=XeniaSystemdRecoveryProfile::bootstrap(grant.clone(),backend,SharedCasStore::default()).unwrap();
    let time=verified_time(&grant,125); let state=verified_state(&grant,&time,vec![]); let (workload,raw)=verified_workload(&grant,&time,0); let proof=verified_xenia(&grant,&time,state,workload,&raw,frontier.head);
    let receipt=profile.recover_verified_once(proof,&time,&plan,ExecutionId("exec-1".into()),ReservationId("reservation-1".into())).unwrap();
    assert_eq!(*calls.lock().unwrap(),1); assert_eq!(receipt.recovery.verification,VerificationResult::Healthy);
}

#[test]
fn process_instance_substitution_fails_before_backend_dispatch() {
    let (before,plan,grant)=plan_and_grant(); let calls=Arc::new(Mutex::new(0));
    let backend=FakeBackend{observations:vec![before].into(),restart_calls:calls.clone()};
    let (mut profile,frontier)=XeniaSystemdRecoveryProfile::bootstrap(grant.clone(),backend,SharedCasStore::default()).unwrap();
    let time=verified_time(&grant,125); let state=verified_state(&grant,&time,vec![]); let (workload,raw)=verified_workload(&grant,&time,1); let proof=verified_xenia(&grant,&time,state,workload,&raw,frontier.head);
    assert!(matches!(profile.recover_verified_once(proof,&time,&plan,ExecutionId("exec-wrong-process".into()),ReservationId("reservation-wrong-process".into())),Err(ProfileRecoveryError::Workload(_))));
    assert_eq!(*calls.lock().unwrap(),0);
}

#[test]
fn authenticated_revocation_is_denied_by_final_broker() {
    let (before,plan,grant)=plan_and_grant(); let calls=Arc::new(Mutex::new(0));
    let backend=FakeBackend{observations:vec![before].into(),restart_calls:calls.clone()};
    let (mut profile,frontier)=XeniaSystemdRecoveryProfile::bootstrap(grant.clone(),backend,SharedCasStore::default()).unwrap();
    let time=verified_time(&grant,125); let state=verified_state(&grant,&time,vec![NegativeAuthorityFact::RevokeGrant{grant_digest:grant.digest()}]); let (workload,raw)=verified_workload(&grant,&time,0); let proof=verified_xenia(&grant,&time,state,workload,&raw,frontier.head);
    assert!(matches!(profile.recover_verified_once(proof,&time,&plan,ExecutionId("exec-revoked".into()),ReservationId("reservation-revoked".into())),Err(ProfileRecoveryError::Broker(_))));
    assert_eq!(*calls.lock().unwrap(),0);
}
