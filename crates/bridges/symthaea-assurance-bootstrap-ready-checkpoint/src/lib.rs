// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Independently verified signed checkpoint-one token for launch-bound bootstrap.
//!
//! The builder consumes the opaque launch-bound first live observation, so it
//! can establish that checkpoint one commits that exact observation. The wire
//! verifier deliberately cannot make that claim from bytes alone; it proves
//! the reviewed runtime policy, signed launch, signed checkpoint-one lineage,
//! and the claimed dynamic-measurement digest. A later composition can rebind
//! those exact bytes to the opaque builder capability.

#![deny(unsafe_code)]

use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use symthaea_assurance_launch_bound_first_runtime_observation::LaunchBoundFirstRuntimeObservation;
use symthaea_evidence_verifier_runtime_continuity::{
    RuntimeMeasurementCheckpoint, RuntimeMeasurementScope, VerifierRuntimeContinuityPolicy,
    VerifierRuntimeLaunchAttestation,
};

pub const BOOTSTRAP_READY_POLICY_SCHEMA_V1: &str =
    "symthaea.assurance.bootstrap-ready-checkpoint-policy.v1";
pub const BOOTSTRAP_READY_WIRE_SCHEMA_V1: &str =
    "symthaea.assurance.bootstrap-ready-checkpoint-wire.v1";
pub const BOOTSTRAP_READY_REPORT_SCHEMA_V1: &str =
    "symthaea.assurance.bootstrap-ready-checkpoint-report.v1";
const POLICY_DOMAIN: &[u8] = b"symthaea.assurance.bootstrap-ready-checkpoint-policy.digest.v1\0";
const WIRE_DOMAIN: &[u8] = b"symthaea.assurance.bootstrap-ready-checkpoint-wire.digest.v1\0";
const REPORT_DOMAIN: &[u8] = b"symthaea.assurance.bootstrap-ready-checkpoint-report.digest.v1\0";
const QUALIFICATION_DOMAIN: &[u8] =
    b"symthaea.assurance.bootstrap-ready-checkpoint-qualification.digest.v1\0";
const WIRE_MAGIC: &[u8] = b"SYMT-BOOTSTRAP-READY-V1\0";
const MAX_TEXT: usize = 16 * 1024;
const MAX_REFS: usize = 128;
const HARD_MAX_WIRE_BYTES: u32 = 64 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BootstrapReadyCheckpointPolicy {
    pub schema_version: String,
    pub policy_id: String,
    pub expected_launch_bound_policy_digest: String,
    pub expected_runtime_policy_digest: String,
    pub expected_runtime_verifier_ref: String,
    pub max_wire_bytes: u32,
    pub evidence_refs: Vec<String>,
}
impl BootstrapReadyCheckpointPolicy {
    pub fn validate(&self) -> bool {
        self.schema_version == BOOTSTRAP_READY_POLICY_SCHEMA_V1
            && text(&self.policy_id)
            && digest(&self.expected_launch_bound_policy_digest)
            && digest(&self.expected_runtime_policy_digest)
            && text(&self.expected_runtime_verifier_ref)
            && (1..=HARD_MAX_WIRE_BYTES).contains(&self.max_wire_bytes)
            && refs(&self.evidence_refs)
    }
    pub fn canonical_digest(&self) -> Option<String> {
        if !self.validate() { return None; }
        let mut h = blake3::Hasher::new();
        h.update(POLICY_DOMAIN);
        for v in [
            self.schema_version.as_str(), self.policy_id.as_str(),
            self.expected_launch_bound_policy_digest.as_str(),
            self.expected_runtime_policy_digest.as_str(),
            self.expected_runtime_verifier_ref.as_str(),
        ] { field(&mut h, v); }
        h.update(&self.max_wire_bytes.to_le_bytes());
        sorted(&mut h, &self.evidence_refs);
        Some(b3(h.finalize()))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BootstrapReadyWire {
    pub schema_version: String,
    pub runtime_policy_digest: String,
    pub launch_bound_policy_digest: String,
    pub launch_bound_qualification_digest: String,
    pub ticket_digest: String,
    pub launch_challenge_nonce_blake3_hex: String,
    pub first_observation_qualification_digest: String,
    pub checkpoint_challenge_digest: String,
    pub process_instance_id: String,
    pub checkpoint_monotonic_counter: u64,
    pub observed_at_ms: u64,
    pub launch: VerifierRuntimeLaunchAttestation,
    pub checkpoint: RuntimeMeasurementCheckpoint,
}
impl BootstrapReadyWire {
    pub fn validate_shape(&self) -> bool {
        self.schema_version == BOOTSTRAP_READY_WIRE_SCHEMA_V1
            && digest(&self.runtime_policy_digest)
            && digest(&self.launch_bound_policy_digest)
            && digest(&self.launch_bound_qualification_digest)
            && digest(&self.ticket_digest)
            && hex_n(&self.launch_challenge_nonce_blake3_hex, 64)
            && digest(&self.first_observation_qualification_digest)
            && digest(&self.checkpoint_challenge_digest)
            && text(&self.process_instance_id)
            && self.checkpoint_monotonic_counter > 0
            && self.launch.validate()
            && self.checkpoint.validate()
    }
    pub fn canonical_bytes(&self) -> Option<Vec<u8>> {
        if !self.validate_shape() { return None; }
        let mut out = WIRE_MAGIC.to_vec();
        for v in [
            self.schema_version.as_str(), self.runtime_policy_digest.as_str(),
            self.launch_bound_policy_digest.as_str(), self.launch_bound_qualification_digest.as_str(),
            self.ticket_digest.as_str(), self.launch_challenge_nonce_blake3_hex.as_str(),
            self.first_observation_qualification_digest.as_str(), self.checkpoint_challenge_digest.as_str(),
            self.process_instance_id.as_str(),
        ] { wstr(&mut out, v)?; }
        out.extend_from_slice(&self.checkpoint_monotonic_counter.to_le_bytes());
        out.extend_from_slice(&self.observed_at_ms.to_le_bytes());
        enc_launch(&mut out, &self.launch)?;
        enc_checkpoint(&mut out, &self.checkpoint)?;
        Some(out)
    }
    pub fn canonical_digest(&self) -> Option<String> {
        let bytes = self.canonical_bytes()?;
        let mut h = blake3::Hasher::new();
        h.update(WIRE_DOMAIN); h.update(&(bytes.len() as u64).to_le_bytes()); h.update(&bytes);
        Some(b3(h.finalize()))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BootstrapReadyDisposition { Invalid, Blocked, Qualified }
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BootstrapReadyIssue {
    InvalidPolicy, InvalidRuntimePolicy, InvalidWire,
    WireTooLarge { observed: u64, maximum: u64 },
    LaunchBoundPolicyMismatch, RuntimePolicyMismatch, RuntimeVerifierMismatch,
    LaunchPolicyMismatch, LaunchVerifierMismatch, LaunchProcessMismatch,
    LaunchBootMeasurementMismatch, LaunchExecutableMismatch, LaunchClosureMismatch,
    LaunchRuntimeConfigMismatch, LaunchOutsidePolicyWindow, LaunchSignatureUntrusted,
    CheckpointLaunchMismatch, CheckpointVerifierMismatch, CheckpointProcessMismatch,
    CheckpointSequenceMismatch, CheckpointPredecessorPresent,
    CheckpointCounterMismatch, CheckpointCounterNotAfterLaunch,
    CheckpointObservationTimeMismatch, CheckpointTimeRegression, CheckpointGapExceeded,
    CheckpointBootMeasurementMismatch, CheckpointExecutableMismatch, CheckpointClosureMismatch,
    CheckpointRuntimeConfigMismatch, CheckpointDynamicMeasurementMismatch,
    CheckpointOutsidePolicyWindow, CheckpointSignatureUntrusted,
    OpaqueLaunchBoundPolicyMismatch, OpaqueLaunchBoundQualificationMismatch,
    OpaqueTicketMismatch, OpaqueChallengeNonceMismatch, OpaqueCheckpointChallengeMismatch,
    OpaqueProcessMismatch, OpaqueCounterMismatch, OpaqueObservationTimeMismatch,
    OpaqueVerifierMismatch, OpaqueExecutableMismatch, OpaqueObservationQualificationMismatch,
}
impl BootstrapReadyIssue {
    fn invalid(&self) -> bool { matches!(self, Self::InvalidPolicy | Self::InvalidRuntimePolicy | Self::InvalidWire) }
    fn code(&self) -> String {
        match self {
            Self::InvalidPolicy=>"invalid-policy".into(), Self::InvalidRuntimePolicy=>"invalid-runtime-policy".into(),
            Self::InvalidWire=>"invalid-wire".into(), Self::WireTooLarge{observed,maximum}=>format!("wire-too-large:{observed}:{maximum}"),
            Self::LaunchBoundPolicyMismatch=>"launch-bound-policy-mismatch".into(), Self::RuntimePolicyMismatch=>"runtime-policy-mismatch".into(),
            Self::RuntimeVerifierMismatch=>"runtime-verifier-mismatch".into(), Self::LaunchPolicyMismatch=>"launch-policy-mismatch".into(),
            Self::LaunchVerifierMismatch=>"launch-verifier-mismatch".into(), Self::LaunchProcessMismatch=>"launch-process-mismatch".into(),
            Self::LaunchBootMeasurementMismatch=>"launch-boot-measurement-mismatch".into(), Self::LaunchExecutableMismatch=>"launch-executable-mismatch".into(),
            Self::LaunchClosureMismatch=>"launch-closure-mismatch".into(), Self::LaunchRuntimeConfigMismatch=>"launch-runtime-config-mismatch".into(),
            Self::LaunchOutsidePolicyWindow=>"launch-outside-policy-window".into(), Self::LaunchSignatureUntrusted=>"launch-signature-untrusted".into(),
            Self::CheckpointLaunchMismatch=>"checkpoint-launch-mismatch".into(), Self::CheckpointVerifierMismatch=>"checkpoint-verifier-mismatch".into(),
            Self::CheckpointProcessMismatch=>"checkpoint-process-mismatch".into(), Self::CheckpointSequenceMismatch=>"checkpoint-sequence-mismatch".into(),
            Self::CheckpointPredecessorPresent=>"checkpoint-predecessor-present".into(), Self::CheckpointCounterMismatch=>"checkpoint-counter-mismatch".into(),
            Self::CheckpointCounterNotAfterLaunch=>"checkpoint-counter-not-after-launch".into(), Self::CheckpointObservationTimeMismatch=>"checkpoint-observation-time-mismatch".into(),
            Self::CheckpointTimeRegression=>"checkpoint-time-regression".into(), Self::CheckpointGapExceeded=>"checkpoint-gap-exceeded".into(),
            Self::CheckpointBootMeasurementMismatch=>"checkpoint-boot-measurement-mismatch".into(), Self::CheckpointExecutableMismatch=>"checkpoint-executable-mismatch".into(),
            Self::CheckpointClosureMismatch=>"checkpoint-closure-mismatch".into(), Self::CheckpointRuntimeConfigMismatch=>"checkpoint-runtime-config-mismatch".into(),
            Self::CheckpointDynamicMeasurementMismatch=>"checkpoint-dynamic-measurement-mismatch".into(), Self::CheckpointOutsidePolicyWindow=>"checkpoint-outside-policy-window".into(),
            Self::CheckpointSignatureUntrusted=>"checkpoint-signature-untrusted".into(), Self::OpaqueLaunchBoundPolicyMismatch=>"opaque-launch-bound-policy-mismatch".into(),
            Self::OpaqueLaunchBoundQualificationMismatch=>"opaque-launch-bound-qualification-mismatch".into(), Self::OpaqueTicketMismatch=>"opaque-ticket-mismatch".into(),
            Self::OpaqueChallengeNonceMismatch=>"opaque-challenge-nonce-mismatch".into(), Self::OpaqueCheckpointChallengeMismatch=>"opaque-checkpoint-challenge-mismatch".into(),
            Self::OpaqueProcessMismatch=>"opaque-process-mismatch".into(), Self::OpaqueCounterMismatch=>"opaque-counter-mismatch".into(),
            Self::OpaqueObservationTimeMismatch=>"opaque-observation-time-mismatch".into(), Self::OpaqueVerifierMismatch=>"opaque-verifier-mismatch".into(),
            Self::OpaqueExecutableMismatch=>"opaque-executable-mismatch".into(), Self::OpaqueObservationQualificationMismatch=>"opaque-observation-qualification-mismatch".into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BootstrapReadyReport {
    pub schema_version: String, pub policy_id: String, pub policy_digest: Option<String>,
    pub runtime_policy_digest: Option<String>, pub wire_digest: Option<String>,
    pub wire_bytes_digest: Option<String>, pub wire_bytes_len: u64,
    pub launch_digest: Option<String>, pub checkpoint_digest: Option<String>,
    pub launch_bound_qualification_digest: Option<String>,
    pub first_observation_qualification_digest: Option<String>,
    pub process_instance_id: String, pub disposition: BootstrapReadyDisposition,
    pub issues: Vec<BootstrapReadyIssue>,
}
impl BootstrapReadyReport {
    pub fn canonical_digest(&self) -> String {
        let mut h = blake3::Hasher::new(); h.update(REPORT_DOMAIN);
        for v in [self.schema_version.as_str(),self.policy_id.as_str(),self.policy_digest.as_deref().unwrap_or("-"),self.runtime_policy_digest.as_deref().unwrap_or("-"),self.wire_digest.as_deref().unwrap_or("-"),self.wire_bytes_digest.as_deref().unwrap_or("-"),self.launch_digest.as_deref().unwrap_or("-"),self.checkpoint_digest.as_deref().unwrap_or("-"),self.launch_bound_qualification_digest.as_deref().unwrap_or("-"),self.first_observation_qualification_digest.as_deref().unwrap_or("-"),self.process_instance_id.as_str()] { field(&mut h,v); }
        h.update(&self.wire_bytes_len.to_le_bytes()); field(&mut h, match self.disposition { BootstrapReadyDisposition::Invalid=>"invalid",BootstrapReadyDisposition::Blocked=>"blocked",BootstrapReadyDisposition::Qualified=>"qualified" });
        h.update(&(self.issues.len() as u64).to_le_bytes()); for i in &self.issues { field(&mut h,&i.code()); } b3(h.finalize())
    }
    pub const fn grants_physical_authority(&self)->bool{false}
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedBootstrapReadyWire {
    wire_digest:String, wire_bytes_digest:String, runtime_policy_digest:String,
    launch_digest:String, checkpoint_digest:String, launch_bound_policy_digest:String,
    launch_bound_qualification_digest:String, ticket_digest:String,
    launch_challenge_nonce_blake3_hex:String, first_observation_qualification_digest:String,
    checkpoint_challenge_digest:String, process_instance_id:String,
    launch_counter:u64, checkpoint_monotonic_counter:u64, launched_at_ms:u64, observed_at_ms:u64,
}
impl VerifiedBootstrapReadyWire {
    pub fn wire_digest(&self)->&str{&self.wire_digest} pub fn wire_bytes_digest(&self)->&str{&self.wire_bytes_digest}
    pub fn runtime_policy_digest(&self)->&str{&self.runtime_policy_digest} pub fn launch_digest(&self)->&str{&self.launch_digest}
    pub fn checkpoint_digest(&self)->&str{&self.checkpoint_digest} pub fn launch_bound_policy_digest(&self)->&str{&self.launch_bound_policy_digest}
    pub fn launch_bound_qualification_digest(&self)->&str{&self.launch_bound_qualification_digest} pub fn ticket_digest(&self)->&str{&self.ticket_digest}
    pub fn launch_challenge_nonce_blake3_hex(&self)->&str{&self.launch_challenge_nonce_blake3_hex}
    pub fn first_observation_qualification_digest(&self)->&str{&self.first_observation_qualification_digest}
    pub fn checkpoint_challenge_digest(&self)->&str{&self.checkpoint_challenge_digest} pub fn process_instance_id(&self)->&str{&self.process_instance_id}
    pub const fn launch_counter(&self)->u64{self.launch_counter} pub const fn checkpoint_monotonic_counter(&self)->u64{self.checkpoint_monotonic_counter}
    pub const fn launched_at_ms(&self)->u64{self.launched_at_ms} pub const fn observed_at_ms(&self)->u64{self.observed_at_ms}
    pub const fn signed_launch_independently_verified(&self)->bool{true} pub const fn signed_checkpoint_one_independently_verified(&self)->bool{true}
    pub const fn parent_counter_and_gap_rules_reproduced(&self)->bool{true} pub const fn checkpoint_commits_claimed_observation_digest(&self)->bool{true}
    pub const fn actual_live_observation_proven_by_wire_alone(&self)->bool{false} pub const fn trusted_time_established(&self)->bool{false}
    pub const fn grants_physical_authority(&self)->bool{false}
}
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BootstrapReadyCheckpoint { qualification_digest:String, report_digest:String, policy_digest:String, wire:VerifiedBootstrapReadyWire }
impl BootstrapReadyCheckpoint {
    pub fn qualification_digest(&self)->&str{&self.qualification_digest} pub fn report_digest(&self)->&str{&self.report_digest}
    pub fn policy_digest(&self)->&str{&self.policy_digest} pub fn wire(&self)->&VerifiedBootstrapReadyWire{&self.wire}
    pub const fn checkpoint_one_commits_actual_opaque_launch_bound_live_observation(&self)->bool{true}
    pub const fn actual_observation_constructed_after_launch_ticket_consumption(&self)->bool{true}
    pub const fn syscall_confinement_established(&self)->bool{false} pub const fn exclusive_ipc_writer_authority_established(&self)->bool{false}
    pub const fn uninterrupted_exec_to_checkpoint_mapping_continuity_established(&self)->bool{false}
    pub const fn trusted_time_established(&self)->bool{false} pub const fn grants_physical_authority(&self)->bool{false}
}
pub struct BootstrapReadyCheckpointQualification { pub report:BootstrapReadyReport, pub wire_bytes:Vec<u8>, verified_wire:VerifiedBootstrapReadyWire, ready:BootstrapReadyCheckpoint }
impl BootstrapReadyCheckpointQualification { pub fn verified_wire(&self)->&VerifiedBootstrapReadyWire{&self.verified_wire} pub fn ready(&self)->&BootstrapReadyCheckpoint{&self.ready} pub fn into_ready(self)->BootstrapReadyCheckpoint{self.ready} }

pub fn qualify_bootstrap_ready_checkpoint(policy:&BootstrapReadyCheckpointPolicy, first:&LaunchBoundFirstRuntimeObservation, runtime_policy:&VerifierRuntimeContinuityPolicy, launch:&VerifierRuntimeLaunchAttestation, checkpoint:&RuntimeMeasurementCheckpoint)->Result<BootstrapReadyCheckpointQualification,BootstrapReadyReport>{
    let rpd=runtime_policy.canonical_digest();
    let wire=BootstrapReadyWire{schema_version:BOOTSTRAP_READY_WIRE_SCHEMA_V1.into(),runtime_policy_digest:rpd.clone().unwrap_or_else(||"-".into()),launch_bound_policy_digest:first.policy_digest().into(),launch_bound_qualification_digest:first.qualification_digest().into(),ticket_digest:first.ticket_digest().into(),launch_challenge_nonce_blake3_hex:first.launch_challenge_nonce_blake3_hex().into(),first_observation_qualification_digest:first.fresh_observation_qualification_digest().into(),checkpoint_challenge_digest:first.checkpoint_challenge_digest().into(),process_instance_id:first.process_instance_id().into(),checkpoint_monotonic_counter:first.checkpoint_monotonic_counter(),observed_at_ms:first.observed_at_ms(),launch:launch.clone(),checkpoint:checkpoint.clone()};
    let mut report=base_report(policy,rpd,&wire);
    if !policy.validate(){report.issues.push(BootstrapReadyIssue::InvalidPolicy);return Err(finalize(report));}
    if !runtime_policy.validate(){report.issues.push(BootstrapReadyIssue::InvalidRuntimePolicy);return Err(finalize(report));}
    if first.policy_digest()!=policy.expected_launch_bound_policy_digest{report.issues.push(BootstrapReadyIssue::OpaqueLaunchBoundPolicyMismatch);}
    if first.qualification_digest()!=wire.launch_bound_qualification_digest{report.issues.push(BootstrapReadyIssue::OpaqueLaunchBoundQualificationMismatch);}
    if first.ticket_digest()!=wire.ticket_digest{report.issues.push(BootstrapReadyIssue::OpaqueTicketMismatch);}
    if first.launch_challenge_nonce_blake3_hex()!=wire.launch_challenge_nonce_blake3_hex{report.issues.push(BootstrapReadyIssue::OpaqueChallengeNonceMismatch);}
    if first.checkpoint_challenge_digest()!=wire.checkpoint_challenge_digest{report.issues.push(BootstrapReadyIssue::OpaqueCheckpointChallengeMismatch);}
    if first.process_instance_id()!=wire.process_instance_id{report.issues.push(BootstrapReadyIssue::OpaqueProcessMismatch);}
    if first.checkpoint_monotonic_counter()!=wire.checkpoint_monotonic_counter{report.issues.push(BootstrapReadyIssue::OpaqueCounterMismatch);}
    if first.observed_at_ms()!=wire.observed_at_ms{report.issues.push(BootstrapReadyIssue::OpaqueObservationTimeMismatch);}
    if first.runtime_verifier_ref()!=policy.expected_runtime_verifier_ref{report.issues.push(BootstrapReadyIssue::OpaqueVerifierMismatch);}
    if first.executable_digest()!=runtime_policy.expected_executable_digest{report.issues.push(BootstrapReadyIssue::OpaqueExecutableMismatch);}
    if first.fresh_observation_qualification_digest()!=wire.first_observation_qualification_digest{report.issues.push(BootstrapReadyIssue::OpaqueObservationQualificationMismatch);}
    if !report.issues.is_empty(){return Err(finalize(report));}
    let (bytes,verified)=match verify_object(policy,runtime_policy,&wire){Ok(v)=>v,Err(v)=>{report.issues.extend(v);return Err(finalize(report));}};
    fill_success(&mut report,&verified,&bytes,first.qualification_digest(),first.fresh_observation_qualification_digest());
    let report_digest=report.canonical_digest(); let pd=policy.canonical_digest().expect("validated policy");
    let qd=qual_digest(&pd,first.qualification_digest(),verified.wire_digest(),verified.checkpoint_digest(),&report_digest);
    let ready=BootstrapReadyCheckpoint{qualification_digest:qd,report_digest,policy_digest:pd,wire:verified.clone()};
    Ok(BootstrapReadyCheckpointQualification{report,wire_bytes:bytes,verified_wire:verified,ready})
}

pub fn verify_bootstrap_ready_wire(policy:&BootstrapReadyCheckpointPolicy,runtime_policy:&VerifierRuntimeContinuityPolicy,bytes:&[u8])->Result<VerifiedBootstrapReadyWire,BootstrapReadyReport>{
    let mut report=BootstrapReadyReport{schema_version:BOOTSTRAP_READY_REPORT_SCHEMA_V1.into(),policy_id:policy.policy_id.clone(),policy_digest:policy.canonical_digest(),runtime_policy_digest:runtime_policy.canonical_digest(),wire_digest:None,wire_bytes_digest:Some(format!("blake3:{}",blake3::hash(bytes).to_hex())),wire_bytes_len:bytes.len() as u64,launch_digest:None,checkpoint_digest:None,launch_bound_qualification_digest:None,first_observation_qualification_digest:None,process_instance_id:String::new(),disposition:BootstrapReadyDisposition::Invalid,issues:Vec::new()};
    if !policy.validate(){report.issues.push(BootstrapReadyIssue::InvalidPolicy);return Err(finalize(report));}
    if !runtime_policy.validate(){report.issues.push(BootstrapReadyIssue::InvalidRuntimePolicy);return Err(finalize(report));}
    if bytes.len()>policy.max_wire_bytes as usize{report.issues.push(BootstrapReadyIssue::WireTooLarge{observed:bytes.len() as u64,maximum:policy.max_wire_bytes as u64});return Err(finalize(report));}
    let Some(wire)=decode_wire(bytes) else{report.issues.push(BootstrapReadyIssue::InvalidWire);return Err(finalize(report));};
    let (canonical,verified)=match verify_object(policy,runtime_policy,&wire){Ok(v)=>v,Err(v)=>{report.issues.extend(v);return Err(finalize(report));}};
    if canonical!=bytes{report.issues.push(BootstrapReadyIssue::InvalidWire);return Err(finalize(report));}
    fill_success(&mut report,&verified,bytes,&verified.launch_bound_qualification_digest,&verified.first_observation_qualification_digest);
    Ok(verified)
}

fn verify_object(policy:&BootstrapReadyCheckpointPolicy,rp:&VerifierRuntimeContinuityPolicy,wire:&BootstrapReadyWire)->Result<(Vec<u8>,VerifiedBootstrapReadyWire),Vec<BootstrapReadyIssue>>{
    let mut x=Vec::new(); if !wire.validate_shape(){x.push(BootstrapReadyIssue::InvalidWire);return Err(x);} let Some(rpd)=rp.canonical_digest() else{x.push(BootstrapReadyIssue::InvalidRuntimePolicy);return Err(x);};
    if wire.launch_bound_policy_digest!=policy.expected_launch_bound_policy_digest{x.push(BootstrapReadyIssue::LaunchBoundPolicyMismatch);} if wire.runtime_policy_digest!=policy.expected_runtime_policy_digest||wire.runtime_policy_digest!=rpd{x.push(BootstrapReadyIssue::RuntimePolicyMismatch);} if rp.verifier_ref!=policy.expected_runtime_verifier_ref{x.push(BootstrapReadyIssue::RuntimeVerifierMismatch);}
    let l=&wire.launch; let c=&wire.checkpoint; let ld=l.canonical_digest(); let cd=c.canonical_digest();
    if l.policy_digest!=rpd{x.push(BootstrapReadyIssue::LaunchPolicyMismatch);} if l.verifier_ref!=rp.verifier_ref{x.push(BootstrapReadyIssue::LaunchVerifierMismatch);} if l.process_instance_id!=wire.process_instance_id{x.push(BootstrapReadyIssue::LaunchProcessMismatch);}
    if l.boot_measurement_digest!=rp.expected_boot_measurement_digest{x.push(BootstrapReadyIssue::LaunchBootMeasurementMismatch);} if l.executable_digest!=rp.expected_executable_digest{x.push(BootstrapReadyIssue::LaunchExecutableMismatch);} if l.dependency_closure_digest!=rp.expected_dependency_closure_digest{x.push(BootstrapReadyIssue::LaunchClosureMismatch);} if l.runtime_config_digest!=rp.expected_runtime_config_digest{x.push(BootstrapReadyIssue::LaunchRuntimeConfigMismatch);}
    if l.launched_at_ms<rp.issued_at_ms||l.launched_at_ms>=rp.expires_at_ms{x.push(BootstrapReadyIssue::LaunchOutsidePolicyWindow);} if !sig_ok(rp,RuntimeMeasurementScope::Launch,l.launched_at_ms,&l.signer_key_id,&l.signer_public_key_ed25519_hex,l.canonical_unsigned_bytes().as_deref(),&l.signature_ed25519_hex){x.push(BootstrapReadyIssue::LaunchSignatureUntrusted);}
    if ld.as_deref()!=Some(c.launch_attestation_digest.as_str()){x.push(BootstrapReadyIssue::CheckpointLaunchMismatch);} if c.verifier_ref!=rp.verifier_ref{x.push(BootstrapReadyIssue::CheckpointVerifierMismatch);} if c.process_instance_id!=wire.process_instance_id{x.push(BootstrapReadyIssue::CheckpointProcessMismatch);} if c.sequence!=1{x.push(BootstrapReadyIssue::CheckpointSequenceMismatch);} if c.previous_checkpoint_digest.is_some(){x.push(BootstrapReadyIssue::CheckpointPredecessorPresent);}
    if c.monotonic_counter!=wire.checkpoint_monotonic_counter{x.push(BootstrapReadyIssue::CheckpointCounterMismatch);} if c.monotonic_counter<=l.launch_counter{x.push(BootstrapReadyIssue::CheckpointCounterNotAfterLaunch);} if c.observed_at_ms!=wire.observed_at_ms{x.push(BootstrapReadyIssue::CheckpointObservationTimeMismatch);} if c.observed_at_ms<l.launched_at_ms{x.push(BootstrapReadyIssue::CheckpointTimeRegression);} else if c.observed_at_ms-l.launched_at_ms>rp.max_checkpoint_gap_ms{x.push(BootstrapReadyIssue::CheckpointGapExceeded);}
    if c.boot_measurement_digest!=rp.expected_boot_measurement_digest{x.push(BootstrapReadyIssue::CheckpointBootMeasurementMismatch);} if c.executable_digest!=rp.expected_executable_digest{x.push(BootstrapReadyIssue::CheckpointExecutableMismatch);} if c.dependency_closure_digest!=rp.expected_dependency_closure_digest{x.push(BootstrapReadyIssue::CheckpointClosureMismatch);} if c.runtime_config_digest!=rp.expected_runtime_config_digest{x.push(BootstrapReadyIssue::CheckpointRuntimeConfigMismatch);} if c.dynamic_measurement_digest!=wire.first_observation_qualification_digest{x.push(BootstrapReadyIssue::CheckpointDynamicMeasurementMismatch);}
    if c.observed_at_ms<rp.issued_at_ms||c.observed_at_ms>=rp.expires_at_ms{x.push(BootstrapReadyIssue::CheckpointOutsidePolicyWindow);} if !sig_ok(rp,RuntimeMeasurementScope::Checkpoint,c.observed_at_ms,&c.signer_key_id,&c.signer_public_key_ed25519_hex,c.canonical_unsigned_bytes().as_deref(),&c.signature_ed25519_hex){x.push(BootstrapReadyIssue::CheckpointSignatureUntrusted);} if !x.is_empty(){return Err(x);}
    let bytes=wire.canonical_bytes().expect("validated wire"); if bytes.len()>policy.max_wire_bytes as usize{return Err(vec![BootstrapReadyIssue::WireTooLarge{observed:bytes.len() as u64,maximum:policy.max_wire_bytes as u64}]);}
    let wd=wire.canonical_digest().expect("validated wire"); let wbd=format!("blake3:{}",blake3::hash(&bytes).to_hex());
    Ok((bytes,VerifiedBootstrapReadyWire{wire_digest:wd,wire_bytes_digest:wbd,runtime_policy_digest:rpd,launch_digest:ld.expect("valid launch"),checkpoint_digest:cd.expect("valid checkpoint"),launch_bound_policy_digest:wire.launch_bound_policy_digest.clone(),launch_bound_qualification_digest:wire.launch_bound_qualification_digest.clone(),ticket_digest:wire.ticket_digest.clone(),launch_challenge_nonce_blake3_hex:wire.launch_challenge_nonce_blake3_hex.clone(),first_observation_qualification_digest:wire.first_observation_qualification_digest.clone(),checkpoint_challenge_digest:wire.checkpoint_challenge_digest.clone(),process_instance_id:wire.process_instance_id.clone(),launch_counter:l.launch_counter,checkpoint_monotonic_counter:c.monotonic_counter,launched_at_ms:l.launched_at_ms,observed_at_ms:c.observed_at_ms}))
}

fn sig_ok(p:&VerifierRuntimeContinuityPolicy,scope:RuntimeMeasurementScope,at:u64,key_id:&str,pk:&str,msg:Option<&[u8]>,sig:&str)->bool{let Some(msg)=msg else{return false;};let Some(k)=p.trusted_keys.iter().find(|k|k.key_id==key_id)else{return false;};if k.public_key_ed25519_hex!=pk||at<k.valid_from_ms||k.valid_until_ms.is_some_and(|u|at>=u)||k.revoked_at_ms.is_some_and(|r|at>=r)||!k.allowed_scopes.contains(&scope){return false;}let Ok(pkb)=hex::decode(pk)else{return false;};let Ok(sb)=hex::decode(sig)else{return false;};let Ok(pka):Result<[u8;32],_>=pkb.try_into()else{return false;};let Ok(sa):Result<[u8;64],_>=sb.try_into()else{return false;};let Ok(vk)=VerifyingKey::from_bytes(&pka)else{return false;};vk.verify(msg,&Signature::from_bytes(&sa)).is_ok()}

fn fill_success(r:&mut BootstrapReadyReport,v:&VerifiedBootstrapReadyWire,bytes:&[u8],lb:&str,obs:&str){r.wire_digest=Some(v.wire_digest.clone());r.wire_bytes_digest=Some(format!("blake3:{}",blake3::hash(bytes).to_hex()));r.wire_bytes_len=bytes.len() as u64;r.launch_digest=Some(v.launch_digest.clone());r.checkpoint_digest=Some(v.checkpoint_digest.clone());r.launch_bound_qualification_digest=Some(lb.into());r.first_observation_qualification_digest=Some(obs.into());r.process_instance_id=v.process_instance_id.clone();r.disposition=BootstrapReadyDisposition::Qualified;}
fn base_report(p:&BootstrapReadyCheckpointPolicy,rpd:Option<String>,w:&BootstrapReadyWire)->BootstrapReadyReport{BootstrapReadyReport{schema_version:BOOTSTRAP_READY_REPORT_SCHEMA_V1.into(),policy_id:p.policy_id.clone(),policy_digest:p.canonical_digest(),runtime_policy_digest:rpd,wire_digest:w.canonical_digest(),wire_bytes_digest:w.canonical_bytes().map(|b|format!("blake3:{}",blake3::hash(&b).to_hex())),wire_bytes_len:w.canonical_bytes().map(|b|b.len() as u64).unwrap_or(0),launch_digest:w.launch.canonical_digest(),checkpoint_digest:w.checkpoint.canonical_digest(),launch_bound_qualification_digest:Some(w.launch_bound_qualification_digest.clone()),first_observation_qualification_digest:Some(w.first_observation_qualification_digest.clone()),process_instance_id:w.process_instance_id.clone(),disposition:BootstrapReadyDisposition::Invalid,issues:Vec::new()}}
fn finalize(mut r:BootstrapReadyReport)->BootstrapReadyReport{r.disposition=if r.issues.iter().any(BootstrapReadyIssue::invalid){BootstrapReadyDisposition::Invalid}else{BootstrapReadyDisposition::Blocked};r}
fn qual_digest(p:&str,l:&str,w:&str,c:&str,r:&str)->String{let mut h=blake3::Hasher::new();h.update(QUALIFICATION_DOMAIN);for v in[p,l,w,c,r]{field(&mut h,v);}b3(h.finalize())}

fn enc_launch(o:&mut Vec<u8>,v:&VerifierRuntimeLaunchAttestation)->Option<()>{for s in[v.schema_version.as_str(),v.verifier_ref.as_str(),v.process_instance_id.as_str(),v.boot_session_digest.as_str()]{wstr(o,s)?;}o.extend_from_slice(&v.launch_counter.to_le_bytes());for s in[v.policy_digest.as_str(),v.boot_measurement_digest.as_str(),v.executable_digest.as_str(),v.dependency_closure_digest.as_str(),v.runtime_config_digest.as_str()]{wstr(o,s)?;}o.extend_from_slice(&v.launched_at_ms.to_le_bytes());for s in[v.nonce_blake3_hex.as_str(),v.signer_key_id.as_str(),v.signer_public_key_ed25519_hex.as_str(),v.signature_ed25519_hex.as_str()]{wstr(o,s)?;}Some(())}
fn enc_checkpoint(o:&mut Vec<u8>,v:&RuntimeMeasurementCheckpoint)->Option<()>{for s in[v.schema_version.as_str(),v.launch_attestation_digest.as_str(),v.verifier_ref.as_str(),v.process_instance_id.as_str()]{wstr(o,s)?;}o.extend_from_slice(&v.sequence.to_le_bytes());wopt(o,v.previous_checkpoint_digest.as_deref())?;o.extend_from_slice(&v.observed_at_ms.to_le_bytes());o.extend_from_slice(&v.monotonic_counter.to_le_bytes());for s in[v.boot_measurement_digest.as_str(),v.executable_digest.as_str(),v.dependency_closure_digest.as_str(),v.runtime_config_digest.as_str(),v.dynamic_measurement_digest.as_str(),v.signer_key_id.as_str(),v.signer_public_key_ed25519_hex.as_str(),v.signature_ed25519_hex.as_str()]{wstr(o,s)?;}Some(())}
fn decode_wire(b:&[u8])->Option<BootstrapReadyWire>{if !b.starts_with(WIRE_MAGIC){return None;}let mut c=Cursor{b,o:WIRE_MAGIC.len()};let v=BootstrapReadyWire{schema_version:c.s()?,runtime_policy_digest:c.s()?,launch_bound_policy_digest:c.s()?,launch_bound_qualification_digest:c.s()?,ticket_digest:c.s()?,launch_challenge_nonce_blake3_hex:c.s()?,first_observation_qualification_digest:c.s()?,checkpoint_challenge_digest:c.s()?,process_instance_id:c.s()?,checkpoint_monotonic_counter:c.u64()?,observed_at_ms:c.u64()?,launch:dec_launch(&mut c)?,checkpoint:dec_checkpoint(&mut c)?};(c.o==b.len()).then_some(v)}
fn dec_launch(c:&mut Cursor<'_>)->Option<VerifierRuntimeLaunchAttestation>{Some(VerifierRuntimeLaunchAttestation{schema_version:c.s()?,verifier_ref:c.s()?,process_instance_id:c.s()?,boot_session_digest:c.s()?,launch_counter:c.u64()?,policy_digest:c.s()?,boot_measurement_digest:c.s()?,executable_digest:c.s()?,dependency_closure_digest:c.s()?,runtime_config_digest:c.s()?,launched_at_ms:c.u64()?,nonce_blake3_hex:c.s()?,signer_key_id:c.s()?,signer_public_key_ed25519_hex:c.s()?,signature_ed25519_hex:c.s()?})}
fn dec_checkpoint(c:&mut Cursor<'_>)->Option<RuntimeMeasurementCheckpoint>{Some(RuntimeMeasurementCheckpoint{schema_version:c.s()?,launch_attestation_digest:c.s()?,verifier_ref:c.s()?,process_instance_id:c.s()?,sequence:c.u64()?,previous_checkpoint_digest:c.opt()?,observed_at_ms:c.u64()?,monotonic_counter:c.u64()?,boot_measurement_digest:c.s()?,executable_digest:c.s()?,dependency_closure_digest:c.s()?,runtime_config_digest:c.s()?,dynamic_measurement_digest:c.s()?,signer_key_id:c.s()?,signer_public_key_ed25519_hex:c.s()?,signature_ed25519_hex:c.s()?})}
struct Cursor<'a>{b:&'a[u8],o:usize}impl Cursor<'_>{fn u32(&mut self)->Option<u32>{let e=self.o.checked_add(4)?;let v=u32::from_le_bytes(self.b.get(self.o..e)?.try_into().ok()?);self.o=e;Some(v)}fn u64(&mut self)->Option<u64>{let e=self.o.checked_add(8)?;let v=u64::from_le_bytes(self.b.get(self.o..e)?.try_into().ok()?);self.o=e;Some(v)}fn s(&mut self)->Option<String>{let n=self.u32()? as usize;if n==0||n>MAX_TEXT{return None;}let e=self.o.checked_add(n)?;let v=std::str::from_utf8(self.b.get(self.o..e)?).ok()?.to_string();self.o=e;text(&v).then_some(v)}fn opt(&mut self)->Option<Option<String>>{let t=*self.b.get(self.o)?;self.o+=1;match t{0=>Some(None),1=>self.s().map(Some),_=>None}}}
fn wstr(o:&mut Vec<u8>,v:&str)->Option<()>{if !text(v)||v.len()>u32::MAX as usize{return None;}o.extend_from_slice(&(v.len() as u32).to_le_bytes());o.extend_from_slice(v.as_bytes());Some(())}
fn wopt(o:&mut Vec<u8>,v:Option<&str>)->Option<()>{match v{Some(v)=>{o.push(1);wstr(o,v)},None=>{o.push(0);Some(())}}}
fn text(v:&str)->bool{!v.is_empty()&&v==v.trim()&&v.len()<=MAX_TEXT&&!v.chars().any(char::is_control)}fn digest(v:&str)->bool{v.strip_prefix("blake3:").is_some_and(|x|hex_n(x,64))}fn hex_n(v:&str,n:usize)->bool{v.len()==n&&v.bytes().all(|b|b.is_ascii_digit()||(b'a'..=b'f').contains(&b))}
fn refs(v:&[String])->bool{v.len()<=MAX_REFS&&v.iter().all(|x|text(x))&&unique(v)}fn unique(v:&[String])->bool{let mut s=BTreeSet::new();v.iter().all(|x|s.insert(x.as_str()))}fn field(h:&mut blake3::Hasher,v:&str){h.update(&(v.len() as u64).to_le_bytes());h.update(v.as_bytes());}fn sorted(h:&mut blake3::Hasher,v:&[String]){let mut x=v.to_vec();x.sort();h.update(&(x.len() as u64).to_le_bytes());for s in x{field(h,&s);}}fn b3(h:blake3::Hash)->String{format!("blake3:{}",h.to_hex())}

#[cfg(test)]mod tests{use super::*;fn d(s:&str)->String{format!("blake3:{}",blake3::hash(s.as_bytes()).to_hex())}fn p()->BootstrapReadyCheckpointPolicy{BootstrapReadyCheckpointPolicy{schema_version:BOOTSTRAP_READY_POLICY_SCHEMA_V1.into(),policy_id:"policy:bootstrap-ready:1".into(),expected_launch_bound_policy_digest:d("lb"),expected_runtime_policy_digest:d("rp"),expected_runtime_verifier_ref:"verifier:host".into(),max_wire_bytes:32768,evidence_refs:vec!["a".into(),"b".into()]}}
#[test]fn policy_ref_order_nonsemantic(){let a=p();let mut b=a.clone();b.evidence_refs.reverse();assert_eq!(a.canonical_digest(),b.canonical_digest());b.expected_runtime_verifier_ref="other".into();assert_ne!(a.canonical_digest(),b.canonical_digest());}
#[test]fn optional_wire_encoding_unambiguous(){let mut a=Vec::new();wopt(&mut a,None).unwrap();let mut b=Vec::new();wopt(&mut b,Some("x")).unwrap();assert_ne!(a,b);}
#[test]fn claim_ceiling(){let c=["builder_actual_live=true","wire_actual_live=false","parent_counter_gap=true","syscall_confinement=false","exclusive_writer=false","continuous_mapping=false","trusted_time=false","physical_authority=false"];assert_eq!(c.len(),8);}}
