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
use symthaea_system_attempt_evidence::{AttemptEvidenceJournal, AttemptEvidenceState, SqliteAttemptEvidenceJournal};
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
use symthaea_xenia_systemd_durable_profile::{
    DurableAttemptEvidenceStatus, DurableRecoveryError, DurableXeniaSystemdBootstrap,
};

static NEXT_DB: AtomicU64 = AtomicU64::new(0);
#[derive(Debug, Clone, Copy)] struct FakeBackendError;
impl std::fmt::Display for FakeBackendError { fn fmt(&self,f:&mut std::fmt::Formatter<'_>)->std::fmt::Result { f.write_str("fake backend") } }
impl std::error::Error for FakeBackendError {}
struct FakeBackend { observations: VecDeque<ServiceObservation>, restart_calls: Arc<Mutex<usize>> }
impl ServiceBackend for FakeBackend {
    type Error=FakeBackendError;
    fn observe(&mut self,_:&HostId,_:&ServiceUnit)->Result<ServiceObservation,Self::Error>{self.observations.pop_front().ok_or(FakeBackendError)}
    fn restart(&mut self,_:&HostId,_:&ServiceUnit)->Result<DispatchEvidence,Self::Error>{*self.restart_calls.lock().map_err(|_|FakeBackendError)?+=1;Ok(DispatchEvidence::Applied)}
}
fn db_path()->std::path::PathBuf{let id=NEXT_DB.fetch_add(1,Ordering::Relaxed);std::env::temp_dir().join(format!("symthaea-measured-durable-{}-{id}.sqlite",std::process::id()))}
fn cleanup(path:&std::path::Path){let _=fs::remove_file(path);let _=fs::remove_file(format!("{}-wal",path.display()));let _=fs::remove_file(format!("{}-shm",path.display()));}
fn host()->HostId{HostId::parse("host-a").unwrap()} fn unit()->ServiceUnit{ServiceUnit::parse("postgresql.service").unwrap()}
fn obs(active:&str,sub:&str,inv:&str)->ServiceObservation{ServiceObservation{host:host(),unit:unit(),active_state:active.into(),sub_state:sub.into(),invocation_id:Some(inv.into())}}
fn plan_grant()->(ServiceObservation,RestartPlan,CapabilityGrant){let before=obs("failed","failed","old");let actor=PrincipalId("symthaea://agent/system-recovery".into());let executor=PrincipalId("spiffe://luminous.local/symthaea/system-broker".into());let task=Some(TaskId("repair-postgresql".into()));let plan=RestartPlan::new(actor.clone(),executor.clone(),task.clone(),&before);let mut grant=CapabilityGrant::new("durable-grant",PrincipalId("xenia://operator/alice".into()),actor,AuthorityEpoch(7));grant.audience=Some(executor);grant.task=task;grant.resources=BTreeSet::from([plan.resource()]);grant.operations=BTreeSet::from([plan.operation()]);grant.plan_digest=Some(plan.digest());grant.world_digest=Some(plan.world_digest);grant.expires_at_unix_s=Some(180);grant.max_uses=1;grant.risk_budget=restart_risk_charge();(before,plan,grant)}

fn time(grant:&CapabilityGrant,w:u64)->VerifiedAuthorityTime{let a=SigningKey::from_bytes(&[31;32]);let b=SigningKey::from_bytes(&[32;32]);let p=TrustedTimePolicyV1{schema_version:AUTHORITY_TIME_SCHEMA_VERSION,policy_id:[33;16],authorities:vec![TrustedTimeAuthorityV1{authority_id:TimeAuthorityId([1;16]),verifying_key:a.verifying_key().to_bytes(),organization_binding:[41;32],service_binding:[51;32]},TrustedTimeAuthorityV1{authority_id:TimeAuthorityId([2;16]),verifying_key:b.verifying_key().to_bytes(),organization_binding:[42;32],service_binding:[52;32]}],threshold:2,minimum_organizations:2,maximum_uncertainty_s:1,maximum_challenge_age_ns:5_000_000_000,maximum_post_verification_age_ns:5_000_000_000};let pending=PendingAuthorityTimeChallenge::new(&p,grant.digest().0).unwrap();let c=pending.wire();let sign=|id:TimeAuthorityId,k:&SigningKey|{let mut s=AuthorityTimeStatementV1{schema_version:1,authority_id:id,policy_digest:c.policy_digest,subject_digest:c.subject_digest,challenge_nonce:c.nonce,witnessed_unix_s:w,uncertainty_s:1,signature:vec![]};s.signature=k.sign(&s.canonical_message().unwrap()).to_bytes().to_vec();s};verify_authority_time_v1(&p,pending,&[sign(TimeAuthorityId([1;16]),&a),sign(TimeAuthorityId([2;16]),&b)]).unwrap()}
fn state(grant:&CapabilityGrant,t:&VerifiedAuthorityTime)->VerifiedAuthorityState{let a=SigningKey::from_bytes(&[61;32]);let b=SigningKey::from_bytes(&[62;32]);let p=AuthorityStatePolicyV1{schema_version:AUTHORITY_STATE_SCHEMA_VERSION,policy_id:[63;16],witnesses:vec![TrustedAuthorityStateWitnessV1{witness_id:AuthorityStateWitnessId([1;16]),verifying_key:a.verifying_key().to_bytes(),organization_binding:[71;32],service_binding:[81;32]},TrustedAuthorityStateWitnessV1{witness_id:AuthorityStateWitnessId([2;16]),verifying_key:b.verifying_key().to_bytes(),organization_binding:[72;32],service_binding:[82;32]}],threshold:2,minimum_organizations:2,maximum_challenge_age_s:60,maximum_post_verification_age_s:60};let pending=PendingAuthorityStateChallenge::new(&p,grant,t).unwrap();let c=pending.wire();let sign=|id:AuthorityStateWitnessId,k:&SigningKey,g:u64|{let mut s=AuthorityStateStatementV1{schema_version:1,witness_id:id,challenge_nonce:c.nonce,grant_digest:c.grant_digest,state_policy_digest:c.state_policy_digest,time_policy_digest:c.time_policy_digest,source_frontier_sequence:20,source_frontier_digest:Digest32([45;32]),state_sequence:20,authority_epoch:grant.authority_epoch,negative_facts:vec![],witness_generation:g,signature:vec![]};s.signature=k.sign(&s.canonical_message().unwrap()).to_bytes().to_vec();s};verify_authority_state_v1(&p,grant,pending,t,&[sign(AuthorityStateWitnessId([1;16]),&a,1),sign(AuthorityStateWitnessId([2;16]),&b,2)]).unwrap()}
fn workload(grant:&CapabilityGrant,t:&VerifiedAuthorityTime,delta:u64)->(VerifiedExecutorWorkload,ExecutorWorkloadV1){let direct=measure_linux_process_instance(std::process::id()).unwrap();let mut process=direct.process;process.start_time_ticks=process.start_time_ticks.saturating_add(delta);let raw=ExecutorWorkloadV1{executor:grant.audience.clone().unwrap(),artifact_digest:direct.artifact_digest,configuration_digest:Digest32([91;32]),host_identity_digest:direct.host_identity_digest};let a=SigningKey::from_bytes(&[101;32]);let b=SigningKey::from_bytes(&[102;32]);let p=WorkloadWitnessPolicyV1{schema_version:EXECUTOR_WORKLOAD_SCHEMA_VERSION,policy_id:[103;16],witnesses:vec![TrustedWorkloadWitnessV1{witness_id:WorkloadWitnessId([1;16]),verifying_key:a.verifying_key().to_bytes(),organization_binding:[111;32],service_binding:[121;32]},TrustedWorkloadWitnessV1{witness_id:WorkloadWitnessId([2;16]),verifying_key:b.verifying_key().to_bytes(),organization_binding:[112;32],service_binding:[122;32]}],threshold:2,minimum_organizations:2,maximum_challenge_age_s:10,maximum_post_verification_age_s:10,require_nix_store_executable:false};let pending=PendingWorkloadChallenge::new(&p,grant,t).unwrap();let c=pending.wire();let sign=|id:WorkloadWitnessId,k:&SigningKey,g:u64|{let mut s=WorkloadMeasurementStatementV1{schema_version:1,witness_id:id,challenge_nonce:c.nonce,grant_digest:c.grant_digest,executor:c.executor.clone(),workload_policy_digest:c.workload_policy_digest,time_policy_digest:c.time_policy_digest,workload:raw.clone(),process,witness_generation:g,executable_in_nix_store:direct.executable_in_nix_store,signature:vec![]};s.signature=k.sign(&s.canonical_message().unwrap()).to_bytes().to_vec();s};let v=verify_executor_workload_v1(&p,grant,pending,t,&[sign(WorkloadWitnessId([1;16]),&a,1),sign(WorkloadWitnessId([2;16]),&b,2)]).unwrap();(v,raw)}
fn proof(grant:&CapabilityGrant,t:&VerifiedAuthorityTime,s:VerifiedAuthorityState,w:VerifiedExecutorWorkload,raw:&ExecutorWorkloadV1,head:CheckpointHead)->symthaea_xenia_authority::VerifiedXeniaCapability{let k=SigningKey::from_bytes(&[3;32]);let pk=k.verifying_key().to_bytes();let auth=XeniaAgentAuthorizationV1{schema_version:1,authorization_id:[41;16],session_id:[42;16],session_transcript_hash:[43;32],session_signature_suite:TranscriptSignatureSuiteV1::Ed25519Rfc8032,capability_digest:grant.digest().0,executor_workload_digest:raw.digest().unwrap().0,authority_epoch:grant.authority_epoch.0,issued_at_unix_s:100,expires_at_unix_s:160,nonce:[44;16],ledger_entry_count:20,ledger_head_hash:[45;32],prior_checkpoint:Some(XeniaCheckpointAnchorV1{sequence:head.sequence,digest:head.digest.0})};let att=XeniaAgentCapabilityAttestationV1{schema:AGENT_CAPABILITY_ATTESTATION_SCHEMA.into(),ledger_public_key_fingerprint:*blake3::hash(&pk).as_bytes(),signature:XeniaSignatureEnvelopeV1{algorithm:ED25519_SIGNATURE_ALGORITHM.into(),signature:k.sign(&auth.canonical_message().unwrap()).to_bytes().to_vec()},authorization:auth};let mut cp=XeniaLedgerCheckpointV1{schema:XENIA_LEDGER_CHECKPOINT_SCHEMA.into(),entry_count:20,head_hash:[45;32],ledger_public_key:pk,timestamp_unix_secs:120,signature:vec![]};cp.signature=k.sign(&cp.signature_message().unwrap()).to_bytes().to_vec();verify_xenia_capability_v1(&att,&cp,pk,grant,w,XeniaSessionExpectationV1{session_id:[42;16],transcript_hash:[43;32],transcript_signature_suite:TranscriptSignatureSuiteV1::Ed25519Rfc8032},head,t,s,XeniaFreshnessPolicyV1::strict(30,5)).unwrap()}

#[test]
fn durable_success_records_measured_process_before_effect_lineage(){let path=db_path();cleanup(&path);let(before,plan,grant)=plan_grant();let calls=Arc::new(Mutex::new(0));let backend=FakeBackend{observations:vec![before.clone(),before,obs("active","running","new")].into(),restart_calls:calls.clone()};let bootstrap=DurableXeniaSystemdBootstrap::bootstrap(grant.clone(),backend,&path).unwrap();let head=bootstrap.authorization_checkpoint_head();let t=time(&grant,125);let s=state(&grant,&t);let(w,raw)=workload(&grant,&t,0);let p=proof(&grant,&t,s,w,&raw,head);let receipt=bootstrap.recover_verified_once(p,&t,&plan,ExecutionId("exec".into()),ReservationId("res".into())).unwrap();assert_eq!(*calls.lock().unwrap(),1);assert_eq!(receipt.recovery.verification,VerificationResult::Healthy);assert!(matches!(receipt.attempt_evidence,DurableAttemptEvidenceStatus::RecoveryCompleted(_)));let journal=SqliteAttemptEvidenceJournal::open(&path).unwrap();let chain=journal.load_chain(receipt.attempt_key).unwrap();assert_eq!(chain.len(),3);assert_eq!(chain[0].state,AttemptEvidenceState::DispatchArmed);assert_eq!(chain[1].state,AttemptEvidenceState::Applied);assert_eq!(chain[2].state,AttemptEvidenceState::RecoveryCompleted);let store=SqliteCheckpointCasStore::open(&path).unwrap();let(_,durable)=store.load_frontier().unwrap().unwrap();assert_eq!(durable,receipt.recovery.checkpoint_head);cleanup(&path)}

#[test]
fn process_mismatch_fails_before_attempt_row_or_backend(){let path=db_path();cleanup(&path);let(before,plan,grant)=plan_grant();let calls=Arc::new(Mutex::new(0));let backend=FakeBackend{observations:vec![before].into(),restart_calls:calls.clone()};let bootstrap=DurableXeniaSystemdBootstrap::bootstrap(grant.clone(),backend,&path).unwrap();let head=bootstrap.authorization_checkpoint_head();let t=time(&grant,125);let s=state(&grant,&t);let(w,raw)=workload(&grant,&t,1);let p=proof(&grant,&t,s,w,&raw,head);assert!(matches!(bootstrap.recover_verified_once(p,&t,&plan,ExecutionId("bad".into()),ReservationId("bad".into())),Err(DurableRecoveryError::Workload(_))));assert_eq!(*calls.lock().unwrap(),0);let count:i64=rusqlite::Connection::open(&path).unwrap().query_row("SELECT COUNT(*) FROM system_attempt_evidence",[],|r|r.get(0)).unwrap_or(0);assert_eq!(count,0);cleanup(&path)}
