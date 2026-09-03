// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fresh, process-instance-bound executor workload identity.
//!
//! A workload digest is not useful security evidence if a caller can freely
//! construct it. V1 therefore requires fresh independent witness statements for
//! one exact capability grant and executor process instance. Witnesses measure
//! the process they currently observe; the challenge does not contain a
//! caller-selected artifact/configuration/host digest.
//!
//! The resulting [`VerifiedExecutorWorkload`] is opaque and non-cloneable. It
//! can also re-check the local Linux process instance immediately before effect
//! admission. V1 deliberately trusts the host kernel's `/proc` view and witness
//! services; TPM/IMA/fs-verity evidence can strengthen that root later without
//! changing the stable [`ExecutorWorkloadV1`] commitment consumed by Xenia.

#![deny(unsafe_code)]

use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;

use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use symthaea_authority::{CapabilityGrant, Digest32, PrincipalId};
use symthaea_authority_time::{AuthorityTimeError, VerifiedAuthorityTime};
use thiserror::Error;

pub const EXECUTOR_WORKLOAD_SCHEMA_VERSION: u16 = 1;
pub const MAX_WORKLOAD_WITNESSES: usize = 64;
pub const MAX_WORKLOAD_STATEMENTS: usize = 128;
pub const MAX_WORKLOAD_CHALLENGE_AGE_S: u64 = 60;
pub const MAX_EXECUTOR_ID_BYTES: usize = 1024;

const WORKLOAD_DOMAIN: &[u8] = b"symthaea.executor-workload.v1\0";
const POLICY_DOMAIN: &[u8] = b"symthaea.executor-workload.policy.v1\0";
const CHALLENGE_DOMAIN: &[u8] = b"symthaea.executor-workload.challenge.v1\0";
const STATEMENT_DOMAIN: &[u8] = b"symthaea.executor-workload.statement.v1\0";
const PROCESS_DOMAIN: &[u8] = b"symthaea.executor-workload.process.v1\0";

/// Stable security-relevant workload identity committed by Xenia.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutorWorkloadV1 {
    pub executor: PrincipalId,
    pub artifact_digest: Digest32,
    pub configuration_digest: Digest32,
    pub host_identity_digest: Digest32,
}

impl ExecutorWorkloadV1 {
    pub fn validate(&self) -> Result<(), WorkloadIdentityError> {
        if self.executor.0.is_empty() || self.executor.0.len() > MAX_EXECUTOR_ID_BYTES {
            return Err(WorkloadIdentityError::InvalidExecutor);
        }
        if self.artifact_digest.0 == [0; 32] {
            return Err(WorkloadIdentityError::ZeroArtifactDigest);
        }
        if self.configuration_digest.0 == [0; 32] {
            return Err(WorkloadIdentityError::ZeroConfigurationDigest);
        }
        if self.host_identity_digest.0 == [0; 32] {
            return Err(WorkloadIdentityError::ZeroHostIdentityDigest);
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<Digest32, WorkloadIdentityError> {
        self.validate()?;
        let mut transcript = Transcript::new(WORKLOAD_DOMAIN);
        transcript.bytes(self.executor.0.as_bytes())?;
        transcript.fixed(&self.artifact_digest.0);
        transcript.fixed(&self.configuration_digest.0);
        transcript.fixed(&self.host_identity_digest.0);
        Ok(Digest32(transcript.finish()))
    }
}

/// Exact Linux process instance witnesses claim they measured.
///
/// PID alone is reusable. V1 therefore binds PID to kernel boot identity and
/// `/proc/<pid>/stat` process start ticks, plus the resolved executable path.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct LinuxProcessInstanceV1 {
    pub pid: u32,
    pub start_time_ticks: u64,
    pub boot_id_digest: Digest32,
    pub executable_path_digest: Digest32,
}

impl LinuxProcessInstanceV1 {
    pub fn validate(&self) -> Result<(), WorkloadIdentityError> {
        if self.pid == 0 || self.start_time_ticks == 0 {
            return Err(WorkloadIdentityError::InvalidProcessInstance);
        }
        if self.boot_id_digest.0 == [0; 32] || self.executable_path_digest.0 == [0; 32] {
            return Err(WorkloadIdentityError::InvalidProcessInstance);
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<Digest32, WorkloadIdentityError> {
        self.validate()?;
        let mut transcript = Transcript::new(PROCESS_DOMAIN);
        transcript.u32(self.pid);
        transcript.u64(self.start_time_ticks);
        transcript.fixed(&self.boot_id_digest.0);
        transcript.fixed(&self.executable_path_digest.0);
        Ok(Digest32(transcript.finish()))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct WorkloadWitnessId(pub [u8; 16]);

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TrustedWorkloadWitnessV1 {
    pub witness_id: WorkloadWitnessId,
    pub verifying_key: [u8; 32],
    pub organization_binding: [u8; 32],
    pub service_binding: [u8; 32],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkloadWitnessPolicyV1 {
    pub schema_version: u16,
    pub policy_id: [u8; 16],
    pub witnesses: Vec<TrustedWorkloadWitnessV1>,
    pub threshold: u16,
    pub minimum_organizations: u16,
    pub maximum_challenge_age_s: u64,
    pub maximum_post_verification_age_s: u64,
    /// Require the measured executable path itself to resolve under `/nix/store`.
    pub require_nix_store_executable: bool,
}

impl WorkloadWitnessPolicyV1 {
    pub fn validate(&self) -> Result<(), WorkloadIdentityError> {
        if self.schema_version != EXECUTOR_WORKLOAD_SCHEMA_VERSION
            || self.policy_id == [0; 16]
            || self.witnesses.len() < 2
            || self.witnesses.len() > MAX_WORKLOAD_WITNESSES
            || self.threshold < 2
            || usize::from(self.threshold) > self.witnesses.len()
            || self.minimum_organizations < 2
            || self.minimum_organizations > self.threshold
            || self.maximum_challenge_age_s == 0
            || self.maximum_challenge_age_s > MAX_WORKLOAD_CHALLENGE_AGE_S
            || self.maximum_post_verification_age_s == 0
            || self.maximum_post_verification_age_s > MAX_WORKLOAD_CHALLENGE_AGE_S
        {
            return Err(WorkloadIdentityError::InvalidPolicy);
        }

        let mut ids = BTreeSet::new();
        let mut keys = BTreeSet::new();
        let mut services = BTreeSet::new();
        let mut organizations = BTreeSet::new();
        for witness in &self.witnesses {
            if witness.witness_id.0 == [0; 16]
                || witness.verifying_key == [0; 32]
                || witness.organization_binding == [0; 32]
                || witness.service_binding == [0; 32]
                || VerifyingKey::from_bytes(&witness.verifying_key).is_err()
                || !ids.insert(witness.witness_id)
                || !keys.insert(witness.verifying_key)
                || !services.insert(witness.service_binding)
            {
                return Err(WorkloadIdentityError::InvalidPolicy);
            }
            organizations.insert(witness.organization_binding);
        }
        if organizations.len() < usize::from(self.minimum_organizations) {
            return Err(WorkloadIdentityError::InvalidPolicy);
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<[u8; 32], WorkloadIdentityError> {
        self.validate()?;
        let mut transcript = Transcript::new(POLICY_DOMAIN);
        transcript.u16(self.schema_version);
        transcript.fixed(&self.policy_id);
        transcript.u32(u32::try_from(self.witnesses.len()).map_err(|_| WorkloadIdentityError::Encoding)?);
        for witness in &self.witnesses {
            transcript.fixed(&witness.witness_id.0);
            transcript.fixed(&witness.verifying_key);
            transcript.fixed(&witness.organization_binding);
            transcript.fixed(&witness.service_binding);
        }
        transcript.u16(self.threshold);
        transcript.u16(self.minimum_organizations);
        transcript.u64(self.maximum_challenge_age_s);
        transcript.u64(self.maximum_post_verification_age_s);
        transcript.u8(u8::from(self.require_nix_store_executable));
        Ok(transcript.finish())
    }

    fn witness(&self, id: WorkloadWitnessId) -> Option<&TrustedWorkloadWitnessV1> {
        self.witnesses.iter().find(|w| w.witness_id == id)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkloadChallengeV1 {
    pub schema_version: u16,
    pub nonce: [u8; 32],
    pub grant_digest: Digest32,
    pub executor: PrincipalId,
    pub workload_policy_digest: [u8; 32],
    pub time_policy_digest: [u8; 32],
}

impl WorkloadChallengeV1 {
    pub fn digest(&self) -> Result<[u8; 32], WorkloadIdentityError> {
        let mut transcript = Transcript::new(CHALLENGE_DOMAIN);
        transcript.u16(self.schema_version);
        transcript.fixed(&self.nonce);
        transcript.fixed(&self.grant_digest.0);
        transcript.bytes(self.executor.0.as_bytes())?;
        transcript.fixed(&self.workload_policy_digest);
        transcript.fixed(&self.time_policy_digest);
        Ok(transcript.finish())
    }
}

#[derive(Debug)]
pub struct PendingWorkloadChallenge {
    wire: WorkloadChallengeV1,
    created_not_before_unix_s: u64,
}

impl PendingWorkloadChallenge {
    pub fn new(
        policy: &WorkloadWitnessPolicyV1,
        grant: &CapabilityGrant,
        authority_time: &VerifiedAuthorityTime,
    ) -> Result<Self, WorkloadIdentityError> {
        policy.validate()?;
        let grant_digest = grant.digest();
        authority_time.require_subject(grant_digest.0)?;
        let executor = grant
            .audience
            .clone()
            .ok_or(WorkloadIdentityError::GrantMissingExecutorAudience)?;
        if executor.0.is_empty() || executor.0.len() > MAX_EXECUTOR_ID_BYTES {
            return Err(WorkloadIdentityError::InvalidExecutor);
        }
        let _ = authority_time.conservative_now_unix_s()?;
        let (created_not_before_unix_s, _) = authority_time.interval_at_verification();
        let mut nonce = [0u8; 32];
        getrandom::getrandom(&mut nonce).map_err(|_| WorkloadIdentityError::RandomnessUnavailable)?;
        if nonce == [0; 32] {
            return Err(WorkloadIdentityError::RandomnessUnavailable);
        }
        Ok(Self {
            wire: WorkloadChallengeV1 {
                schema_version: EXECUTOR_WORKLOAD_SCHEMA_VERSION,
                nonce,
                grant_digest,
                executor,
                workload_policy_digest: policy.digest()?,
                time_policy_digest: authority_time.policy_digest(),
            },
            created_not_before_unix_s,
        })
    }

    pub fn wire(&self) -> WorkloadChallengeV1 {
        self.wire.clone()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WorkloadMeasurementStatementV1 {
    pub schema_version: u16,
    pub witness_id: WorkloadWitnessId,
    pub challenge_nonce: [u8; 32],
    pub grant_digest: Digest32,
    pub executor: PrincipalId,
    pub workload_policy_digest: [u8; 32],
    pub time_policy_digest: [u8; 32],
    pub workload: ExecutorWorkloadV1,
    pub process: LinuxProcessInstanceV1,
    pub witness_generation: u64,
    pub executable_in_nix_store: bool,
    pub signature: Vec<u8>,
}

impl WorkloadMeasurementStatementV1 {
    pub fn canonical_message(&self) -> Result<Vec<u8>, WorkloadIdentityError> {
        self.workload.validate()?;
        self.process.validate()?;
        if self.schema_version != EXECUTOR_WORKLOAD_SCHEMA_VERSION
            || self.witness_id.0 == [0; 16]
            || self.challenge_nonce == [0; 32]
            || self.grant_digest.0 == [0; 32]
            || self.executor.0.is_empty()
            || self.executor.0.len() > MAX_EXECUTOR_ID_BYTES
            || self.workload_policy_digest == [0; 32]
            || self.time_policy_digest == [0; 32]
            || self.witness_generation == 0
            || self.workload.executor != self.executor
        {
            return Err(WorkloadIdentityError::InvalidStatement);
        }
        let mut transcript = Transcript::new(STATEMENT_DOMAIN);
        transcript.u16(self.schema_version);
        transcript.fixed(&self.witness_id.0);
        transcript.fixed(&self.challenge_nonce);
        transcript.fixed(&self.grant_digest.0);
        transcript.bytes(self.executor.0.as_bytes())?;
        transcript.fixed(&self.workload_policy_digest);
        transcript.fixed(&self.time_policy_digest);
        transcript.fixed(&self.workload.digest()?.0);
        transcript.fixed(&self.process.digest()?.0);
        transcript.u64(self.witness_generation);
        transcript.u8(u8::from(self.executable_in_nix_store));
        Ok(transcript.into_bytes())
    }
}

/// Opaque proof of one exact freshly witnessed executor process identity.
#[derive(Debug)]
pub struct VerifiedExecutorWorkload {
    grant_digest: Digest32,
    workload: ExecutorWorkloadV1,
    process: LinuxProcessInstanceV1,
    workload_policy_digest: [u8; 32],
    time_policy_digest: [u8; 32],
    verified_not_before_unix_s: u64,
    maximum_post_verification_age_s: u64,
    witness_count: u16,
    organization_count: u16,
    executable_in_nix_store: bool,
}

impl VerifiedExecutorWorkload {
    pub fn grant_digest(&self) -> Digest32 { self.grant_digest }
    pub fn workload(&self) -> &ExecutorWorkloadV1 { &self.workload }
    pub fn workload_digest(&self) -> Result<Digest32, WorkloadIdentityError> { self.workload.digest() }
    pub fn process(&self) -> LinuxProcessInstanceV1 { self.process }
    pub fn workload_policy_digest(&self) -> [u8; 32] { self.workload_policy_digest }
    pub fn time_policy_digest(&self) -> [u8; 32] { self.time_policy_digest }
    pub fn witness_count(&self) -> u16 { self.witness_count }
    pub fn organization_count(&self) -> u16 { self.organization_count }
    pub fn executable_in_nix_store(&self) -> bool { self.executable_in_nix_store }

    pub fn ensure_fresh(
        &self,
        grant: &CapabilityGrant,
        authority_time: &VerifiedAuthorityTime,
    ) -> Result<(), WorkloadIdentityError> {
        if grant.digest() != self.grant_digest {
            return Err(WorkloadIdentityError::GrantMismatch);
        }
        authority_time.require_subject(self.grant_digest.0)?;
        if authority_time.policy_digest() != self.time_policy_digest {
            return Err(WorkloadIdentityError::TimePolicyChanged);
        }
        let now = authority_time.conservative_now_unix_s()?;
        if now.saturating_sub(self.verified_not_before_unix_s) > self.maximum_post_verification_age_s {
            return Err(WorkloadIdentityError::VerifiedWorkloadStale);
        }
        Ok(())
    }

    /// Re-measure the current Linux process at effect entry.
    ///
    /// This proves continuity with the witnessed PID/start-time/boot/executable
    /// instance under the same running kernel. It does not provide hardware-
    /// rooted attestation against a compromised kernel/hypervisor.
    pub fn require_current_process(&self) -> Result<(), WorkloadIdentityError> {
        let current = measure_linux_process_instance(std::process::id())?;
        if current.process != self.process
            || current.artifact_digest != self.workload.artifact_digest
            || current.host_identity_digest != self.workload.host_identity_digest
        {
            return Err(WorkloadIdentityError::CurrentProcessMismatch);
        }
        if self.executable_in_nix_store && !current.executable_in_nix_store {
            return Err(WorkloadIdentityError::ExecutableNotInNixStore);
        }
        Ok(())
    }
}

pub fn verify_executor_workload_v1(
    policy: &WorkloadWitnessPolicyV1,
    grant: &CapabilityGrant,
    challenge: PendingWorkloadChallenge,
    response_time: &VerifiedAuthorityTime,
    statements: &[WorkloadMeasurementStatementV1],
) -> Result<VerifiedExecutorWorkload, WorkloadIdentityError> {
    policy.validate()?;
    if statements.len() < usize::from(policy.threshold) || statements.len() > MAX_WORKLOAD_STATEMENTS {
        return Err(WorkloadIdentityError::InsufficientStatements);
    }
    let grant_digest = grant.digest();
    if challenge.wire.schema_version != EXECUTOR_WORKLOAD_SCHEMA_VERSION
        || challenge.wire.grant_digest != grant_digest
        || challenge.wire.workload_policy_digest != policy.digest()?
        || challenge.wire.nonce == [0; 32]
    {
        return Err(WorkloadIdentityError::InvalidChallenge);
    }
    let expected_executor = grant
        .audience
        .as_ref()
        .ok_or(WorkloadIdentityError::GrantMissingExecutorAudience)?;
    if &challenge.wire.executor != expected_executor {
        return Err(WorkloadIdentityError::ExecutorMismatch);
    }

    response_time.require_subject(grant_digest.0)?;
    if response_time.policy_digest() != challenge.wire.time_policy_digest {
        return Err(WorkloadIdentityError::TimePolicyChanged);
    }
    let response_not_after = response_time.conservative_now_unix_s()?;
    if response_not_after
        .checked_sub(challenge.created_not_before_unix_s)
        .ok_or(WorkloadIdentityError::TimeMovedBackward)?
        > policy.maximum_challenge_age_s
    {
        return Err(WorkloadIdentityError::ChallengeExpired);
    }

    let mut ids = BTreeSet::new();
    let mut organizations = BTreeSet::new();
    let mut services = BTreeSet::new();
    let mut agreed: Option<(ExecutorWorkloadV1, LinuxProcessInstanceV1, bool)> = None;

    for statement in statements {
        let witness = policy
            .witness(statement.witness_id)
            .ok_or(WorkloadIdentityError::UnknownWitness)?;
        if statement.schema_version != EXECUTOR_WORKLOAD_SCHEMA_VERSION
            || statement.challenge_nonce != challenge.wire.nonce
            || statement.grant_digest != grant_digest
            || statement.executor != *expected_executor
            || statement.workload_policy_digest != challenge.wire.workload_policy_digest
            || statement.time_policy_digest != challenge.wire.time_policy_digest
            || statement.workload.executor != *expected_executor
            || statement.witness_generation == 0
            || !ids.insert(statement.witness_id)
        {
            return Err(WorkloadIdentityError::InvalidStatement);
        }
        if policy.require_nix_store_executable && !statement.executable_in_nix_store {
            return Err(WorkloadIdentityError::ExecutableNotInNixStore);
        }
        let sig: [u8; 64] = statement.signature.as_slice().try_into()
            .map_err(|_| WorkloadIdentityError::BadSignatureLength)?;
        let key = VerifyingKey::from_bytes(&witness.verifying_key)
            .map_err(|_| WorkloadIdentityError::InvalidPolicy)?;
        key.verify(&statement.canonical_message()?, &Signature::from_bytes(&sig))
            .map_err(|_| WorkloadIdentityError::BadSignature)?;

        match &agreed {
            None => agreed = Some((statement.workload.clone(), statement.process, statement.executable_in_nix_store)),
            Some((workload, process, nix))
                if workload == &statement.workload
                    && process == &statement.process
                    && *nix == statement.executable_in_nix_store => {}
            Some(_) => return Err(WorkloadIdentityError::WitnessDisagreement),
        }
        organizations.insert(witness.organization_binding);
        services.insert(witness.service_binding);
    }

    if ids.len() < usize::from(policy.threshold)
        || organizations.len() < usize::from(policy.minimum_organizations)
        || services.len() < usize::from(policy.threshold)
    {
        return Err(WorkloadIdentityError::InsufficientDiversity);
    }

    let (workload, process, executable_in_nix_store) =
        agreed.ok_or(WorkloadIdentityError::InsufficientStatements)?;
    workload.validate()?;
    process.validate()?;

    let (verified_not_before_unix_s, _) = response_time.interval_at_verification();
    Ok(VerifiedExecutorWorkload {
        grant_digest,
        workload,
        process,
        workload_policy_digest: challenge.wire.workload_policy_digest,
        time_policy_digest: challenge.wire.time_policy_digest,
        verified_not_before_unix_s,
        maximum_post_verification_age_s: policy.maximum_post_verification_age_s,
        witness_count: u16::try_from(ids.len()).map_err(|_| WorkloadIdentityError::Encoding)?,
        organization_count: u16::try_from(organizations.len()).map_err(|_| WorkloadIdentityError::Encoding)?,
        executable_in_nix_store,
    })
}

/// Minimal direct Linux measurement used for point-of-use continuity checks and
/// by simple local witness implementations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LinuxDirectMeasurementV1 {
    pub process: LinuxProcessInstanceV1,
    pub artifact_digest: Digest32,
    pub host_identity_digest: Digest32,
    pub executable_in_nix_store: bool,
}

pub fn measure_linux_process_instance(pid: u32) -> Result<LinuxDirectMeasurementV1, WorkloadIdentityError> {
    if pid == 0 {
        return Err(WorkloadIdentityError::InvalidProcessInstance);
    }
    let proc_dir = PathBuf::from(format!("/proc/{pid}"));
    let boot_id = fs::read("/proc/sys/kernel/random/boot_id")?;
    let boot_id_digest = Digest32(*blake3::hash(trim_ascii(&boot_id)).as_bytes());
    let stat = fs::read_to_string(proc_dir.join("stat"))?;
    let start_time_ticks = parse_proc_start_time_ticks(&stat)?;
    let executable = fs::read_link(proc_dir.join("exe"))?;
    let executable_text = executable.as_os_str().to_string_lossy();
    let executable_path_digest = Digest32(*blake3::hash(executable_text.as_bytes()).as_bytes());
    let artifact = fs::read(&executable)?;
    let artifact_digest = Digest32(*blake3::hash(&artifact).as_bytes());
    let executable_in_nix_store = executable_text.starts_with("/nix/store/");

    // V1 host identity intentionally binds the kernel boot identity. A future
    // hardware-attested profile can replace/extend this with TPM platform state.
    let host_identity_digest = boot_id_digest;
    Ok(LinuxDirectMeasurementV1 {
        process: LinuxProcessInstanceV1 {
            pid,
            start_time_ticks,
            boot_id_digest,
            executable_path_digest,
        },
        artifact_digest,
        host_identity_digest,
        executable_in_nix_store,
    })
}

fn parse_proc_start_time_ticks(stat: &str) -> Result<u64, WorkloadIdentityError> {
    let close = stat.rfind(')').ok_or(WorkloadIdentityError::MalformedProcStat)?;
    let after = stat.get(close + 1..).ok_or(WorkloadIdentityError::MalformedProcStat)?;
    // Fields after comm begin with field 3 (state). Start time is field 22,
    // therefore index 19 in this suffix.
    let value = after
        .split_whitespace()
        .nth(19)
        .ok_or(WorkloadIdentityError::MalformedProcStat)?;
    value.parse().map_err(|_| WorkloadIdentityError::MalformedProcStat)
}

fn trim_ascii(bytes: &[u8]) -> &[u8] {
    let mut start = 0;
    let mut end = bytes.len();
    while start < end && bytes[start].is_ascii_whitespace() { start += 1; }
    while end > start && bytes[end - 1].is_ascii_whitespace() { end -= 1; }
    &bytes[start..end]
}

struct Transcript { bytes: Vec<u8> }
impl Transcript {
    fn new(domain: &[u8]) -> Self {
        let mut bytes = Vec::with_capacity(256);
        bytes.extend_from_slice(&(domain.len() as u32).to_be_bytes());
        bytes.extend_from_slice(domain);
        Self { bytes }
    }
    fn u8(&mut self, value: u8) { self.bytes.push(value); }
    fn u16(&mut self, value: u16) { self.bytes.extend_from_slice(&value.to_be_bytes()); }
    fn u32(&mut self, value: u32) { self.bytes.extend_from_slice(&value.to_be_bytes()); }
    fn u64(&mut self, value: u64) { self.bytes.extend_from_slice(&value.to_be_bytes()); }
    fn fixed<const N: usize>(&mut self, value: &[u8; N]) { self.bytes.extend_from_slice(value); }
    fn bytes(&mut self, value: &[u8]) -> Result<(), WorkloadIdentityError> {
        let len = u32::try_from(value.len()).map_err(|_| WorkloadIdentityError::Encoding)?;
        self.u32(len);
        self.bytes.extend_from_slice(value);
        Ok(())
    }
    fn into_bytes(self) -> Vec<u8> { self.bytes }
    fn finish(self) -> [u8; 32] { *blake3::hash(&self.bytes).as_bytes() }
}

#[derive(Debug, Error)]
pub enum WorkloadIdentityError {
    #[error("executor workload principal is empty or too long")]
    InvalidExecutor,
    #[error("executor artifact digest must not be zero")]
    ZeroArtifactDigest,
    #[error("executor configuration digest must not be zero")]
    ZeroConfigurationDigest,
    #[error("executor host identity digest must not be zero")]
    ZeroHostIdentityDigest,
    #[error("Linux process instance is invalid")]
    InvalidProcessInstance,
    #[error("workload witness policy is invalid")]
    InvalidPolicy,
    #[error("capability grant lacks an exact executor audience")]
    GrantMissingExecutorAudience,
    #[error("secure workload challenge randomness is unavailable")]
    RandomnessUnavailable,
    #[error("workload challenge is invalid")]
    InvalidChallenge,
    #[error("workload challenge exceeded its maximum worst-case age")]
    ChallengeExpired,
    #[error("trusted time moved backward during workload challenge")]
    TimeMovedBackward,
    #[error("trusted-time policy changed during workload verification")]
    TimePolicyChanged,
    #[error("workload statement references an unknown witness")]
    UnknownWitness,
    #[error("workload statement is malformed or does not bind the challenge")]
    InvalidStatement,
    #[error("workload witness Ed25519 signature must be exactly 64 bytes")]
    BadSignatureLength,
    #[error("workload witness signature verification failed")]
    BadSignature,
    #[error("workload witnesses disagree on the measured process/workload")]
    WitnessDisagreement,
    #[error("workload witness threshold/diversity was not satisfied")]
    InsufficientDiversity,
    #[error("not enough workload witness statements were supplied")]
    InsufficientStatements,
    #[error("measured executor does not match the grant audience")]
    ExecutorMismatch,
    #[error("workload proof belongs to a different capability grant")]
    GrantMismatch,
    #[error("verified workload evidence is stale")]
    VerifiedWorkloadStale,
    #[error("measured executable is not under /nix/store as required by policy")]
    ExecutableNotInNixStore,
    #[error("current Linux process no longer matches the witnessed executor instance")]
    CurrentProcessMismatch,
    #[error("/proc process stat record is malformed")]
    MalformedProcStat,
    #[error("workload canonical encoding failed")]
    Encoding,
    #[error("Linux workload measurement I/O failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("verified authority time failed: {0}")]
    AuthorityTime(#[from] AuthorityTimeError),
}
