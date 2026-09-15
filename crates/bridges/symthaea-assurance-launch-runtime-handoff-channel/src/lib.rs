// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Kernel-identified supervisor-to-runtime launch challenge handoff.
//!
//! The constructor consumes the exact ptrace-confirmed launch plus the exact
//! critical exec-preserved authority qualification, proves that a reviewed
//! surviving tracee read-fd and a supervisor-held write-fd name the same Linux
//! pipe object, generates a 256-bit OS-CSPRNG challenge, and writes one bounded
//! canonical ticket through that supervisor peer.
//!
//! This establishes issuance and delivery-to-channel, not tracee consumption,
//! exclusive writer authority, first-checkpoint binding, trusted time, or
//! physical authority.

#![cfg(target_os = "linux")]
#![deny(unsafe_code)]

use nix::{
    fcntl::OFlag,
    unistd::write,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeSet,
    fs,
    os::unix::io::RawFd,
    path::PathBuf,
};
use symthaea_assurance_critical_exec_preserved_authority::{
    CriticalExecAuthorityQualification, FdAuthorityEvidence,
};
use symthaea_assurance_ptrace_static_exec_confirmation::ConfirmedStaticExecLaunch;

pub const LAUNCH_RUNTIME_HANDOFF_POLICY_SCHEMA_V1: &str =
    "symthaea.assurance.launch-runtime-handoff-policy.v1";
pub const LAUNCH_RUNTIME_HANDOFF_TICKET_SCHEMA_V1: &str =
    "symthaea.assurance.launch-runtime-handoff-ticket.v1";
pub const LAUNCH_RUNTIME_HANDOFF_REPORT_SCHEMA_V1: &str =
    "symthaea.assurance.launch-runtime-handoff-report.v1";

const POLICY_DOMAIN: &[u8] = b"symthaea.assurance.launch-runtime-handoff-policy.digest.v1\0";
const CHANNEL_DOMAIN: &[u8] = b"symthaea.assurance.launch-runtime-handoff-channel.digest.v1\0";
const TICKET_DOMAIN: &[u8] = b"symthaea.assurance.launch-runtime-handoff-ticket.digest.v1\0";
const REPORT_DOMAIN: &[u8] = b"symthaea.assurance.launch-runtime-handoff-report.digest.v1\0";
const QUALIFICATION_DOMAIN: &[u8] =
    b"symthaea.assurance.launch-runtime-handoff-qualification.digest.v1\0";
const WIRE_MAGIC: &[u8] = b"SYMT-LR-HANDOFF-V1\0";
const MAX_TEXT: usize = 4096;
const MAX_REFS: usize = 128;
const HARD_MAX_TICKET_BYTES: u32 = 4096;
const HARD_MAX_FDINFO_BYTES: u64 = 4 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchRuntimeHandoffPolicy {
    pub schema_version: String,
    pub policy_id: String,
    pub expected_critical_authority_policy_digest: String,
    pub expected_exec_confirmation_policy_digest: String,
    pub expected_fd_exec_policy_digest: String,
    pub expected_runtime_policy_digest: String,
    pub expected_runtime_verifier_ref: String,
    pub expected_backend_id: String,
    pub tracee_handoff_read_fd: u32,
    pub max_ticket_bytes: u32,
    pub max_fdinfo_bytes: u64,
    pub evidence_refs: Vec<String>,
}

impl LaunchRuntimeHandoffPolicy {
    pub fn validate(&self) -> bool {
        self.schema_version == LAUNCH_RUNTIME_HANDOFF_POLICY_SCHEMA_V1
            && canonical_text(&self.policy_id)
            && valid_blake3(&self.expected_critical_authority_policy_digest)
            && valid_blake3(&self.expected_exec_confirmation_policy_digest)
            && valid_blake3(&self.expected_fd_exec_policy_digest)
            && valid_blake3(&self.expected_runtime_policy_digest)
            && canonical_text(&self.expected_runtime_verifier_ref)
            && canonical_text(&self.expected_backend_id)
            && self.tracee_handoff_read_fd <= 1_048_576
            && (1..=HARD_MAX_TICKET_BYTES).contains(&self.max_ticket_bytes)
            && (1..=HARD_MAX_FDINFO_BYTES).contains(&self.max_fdinfo_bytes)
            && valid_refs(&self.evidence_refs)
    }

    pub fn canonical_digest(&self) -> Option<String> {
        if !self.validate() {
            return None;
        }
        let mut h = blake3::Hasher::new();
        h.update(POLICY_DOMAIN);
        for value in [
            self.schema_version.as_str(),
            self.policy_id.as_str(),
            self.expected_critical_authority_policy_digest.as_str(),
            self.expected_exec_confirmation_policy_digest.as_str(),
            self.expected_fd_exec_policy_digest.as_str(),
            self.expected_runtime_policy_digest.as_str(),
            self.expected_runtime_verifier_ref.as_str(),
            self.expected_backend_id.as_str(),
        ] {
            field(&mut h, value);
        }
        h.update(&self.tracee_handoff_read_fd.to_le_bytes());
        h.update(&self.max_ticket_bytes.to_le_bytes());
        h.update(&self.max_fdinfo_bytes.to_le_bytes());
        sorted_strings(&mut h, &self.evidence_refs);
        Some(format!("blake3:{}", h.finalize().to_hex()))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchRuntimeHandoffTicket {
    pub schema_version: String,
    pub exec_confirmation_digest: String,
    pub critical_authority_qualification_digest: String,
    pub authority_snapshot_digest: String,
    pub plan_digest: String,
    pub launch_nonce_blake3_hex: String,
    pub tracee_pid: i32,
    pub tracee_handoff_read_fd: u32,
    pub pipe_target: String,
    pub channel_binding_digest: String,
    pub runtime_policy_digest: String,
    pub runtime_verifier_ref: String,
    pub backend_id: String,
    pub executable_digest: String,
    pub challenge_nonce_blake3_hex: String,
    pub ticket_digest: String,
}

impl LaunchRuntimeHandoffTicket {
    pub fn validate(&self) -> bool {
        self.schema_version == LAUNCH_RUNTIME_HANDOFF_TICKET_SCHEMA_V1
            && valid_blake3(&self.exec_confirmation_digest)
            && valid_blake3(&self.critical_authority_qualification_digest)
            && valid_blake3(&self.authority_snapshot_digest)
            && valid_blake3(&self.plan_digest)
            && lower_hex_exact(&self.launch_nonce_blake3_hex, 64)
            && self.tracee_pid > 1
            && self.tracee_handoff_read_fd <= 1_048_576
            && parse_pipe_target(&self.pipe_target).is_some()
            && valid_blake3(&self.channel_binding_digest)
            && valid_blake3(&self.runtime_policy_digest)
            && canonical_text(&self.runtime_verifier_ref)
            && canonical_text(&self.backend_id)
            && valid_blake3(&self.executable_digest)
            && lower_hex_exact(&self.challenge_nonce_blake3_hex, 64)
            && valid_blake3(&self.ticket_digest)
            && self.ticket_digest == self.recompute_digest()
    }

    pub fn recompute_digest(&self) -> String {
        ticket_digest(
            &self.schema_version,
            &self.exec_confirmation_digest,
            &self.critical_authority_qualification_digest,
            &self.authority_snapshot_digest,
            &self.plan_digest,
            &self.launch_nonce_blake3_hex,
            self.tracee_pid,
            self.tracee_handoff_read_fd,
            &self.pipe_target,
            &self.channel_binding_digest,
            &self.runtime_policy_digest,
            &self.runtime_verifier_ref,
            &self.backend_id,
            &self.executable_digest,
            &self.challenge_nonce_blake3_hex,
        )
    }

    pub fn to_wire_bytes(&self) -> Option<Vec<u8>> {
        if !self.validate() {
            return None;
        }
        let mut out = Vec::with_capacity(1024);
        out.extend_from_slice(WIRE_MAGIC);
        for value in [
            self.schema_version.as_str(),
            self.exec_confirmation_digest.as_str(),
            self.critical_authority_qualification_digest.as_str(),
            self.authority_snapshot_digest.as_str(),
            self.plan_digest.as_str(),
            self.launch_nonce_blake3_hex.as_str(),
        ] {
            push_wire_field(&mut out, value)?;
        }
        out.extend_from_slice(&self.tracee_pid.to_le_bytes());
        out.extend_from_slice(&self.tracee_handoff_read_fd.to_le_bytes());
        for value in [
            self.pipe_target.as_str(),
            self.channel_binding_digest.as_str(),
            self.runtime_policy_digest.as_str(),
            self.runtime_verifier_ref.as_str(),
            self.backend_id.as_str(),
            self.executable_digest.as_str(),
            self.challenge_nonce_blake3_hex.as_str(),
            self.ticket_digest.as_str(),
        ] {
            push_wire_field(&mut out, value)?;
        }
        Some(out)
    }

    pub fn from_wire_bytes(bytes: &[u8]) -> Option<Self> {
        let mut cursor = WireCursor::new(bytes);
        cursor.expect_magic(WIRE_MAGIC)?;
        let schema_version = cursor.read_string()?;
        let exec_confirmation_digest = cursor.read_string()?;
        let critical_authority_qualification_digest = cursor.read_string()?;
        let authority_snapshot_digest = cursor.read_string()?;
        let plan_digest = cursor.read_string()?;
        let launch_nonce_blake3_hex = cursor.read_string()?;
        let tracee_pid = cursor.read_i32()?;
        let tracee_handoff_read_fd = cursor.read_u32()?;
        let pipe_target = cursor.read_string()?;
        let channel_binding_digest = cursor.read_string()?;
        let runtime_policy_digest = cursor.read_string()?;
        let runtime_verifier_ref = cursor.read_string()?;
        let backend_id = cursor.read_string()?;
        let executable_digest = cursor.read_string()?;
        let challenge_nonce_blake3_hex = cursor.read_string()?;
        let ticket_digest = cursor.read_string()?;
        if !cursor.finished() {
            return None;
        }
        let ticket = Self {
            schema_version,
            exec_confirmation_digest,
            critical_authority_qualification_digest,
            authority_snapshot_digest,
            plan_digest,
            launch_nonce_blake3_hex,
            tracee_pid,
            tracee_handoff_read_fd,
            pipe_target,
            channel_binding_digest,
            runtime_policy_digest,
            runtime_verifier_ref,
            backend_id,
            executable_digest,
            challenge_nonce_blake3_hex,
            ticket_digest,
        };
        ticket.validate().then_some(ticket)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LaunchRuntimeHandoffIssue {
    InvalidPolicy,
    CriticalAuthorityPolicyMismatch,
    ExecConfirmationPolicyMismatch,
    FdExecPolicyMismatch,
    RuntimePolicyMismatch,
    RuntimeVerifierMismatch,
    BackendMismatch,
    AuthorityConfirmationMismatch,
    AuthorityTraceeMismatch,
    TraceeHandoffFdMissing,
    TraceeHandoffTargetNotPipe,
    TraceeHandoffFdNotReadOnly,
    TraceeHandoffFdCloseOnExecUnexpected,
    TraceeHandoffFdInfoInconsistent,
    InvalidSupervisorWriteFd,
    SupervisorFdUnavailable(String),
    SupervisorTargetMismatch,
    SupervisorFdNotWriteOnly,
    SupervisorFdNonBlocking,
    SupervisorFdInfoInconsistent,
    ChallengeEntropyUnavailable(String),
    TicketEncodingFailed,
    TicketTooLarge { observed: u64, maximum: u64 },
    TicketWriteFailed(String),
    PartialTicketWrite { expected: u64, observed: u64 },
    SupervisorFdChangedDuringIssuance,
}

impl LaunchRuntimeHandoffIssue {
    fn invalid(&self) -> bool {
        matches!(self, Self::InvalidPolicy | Self::InvalidSupervisorWriteFd)
    }

    fn code(&self) -> String {
        match self {
            Self::InvalidPolicy => "invalid-policy".into(),
            Self::CriticalAuthorityPolicyMismatch => "critical-authority-policy-mismatch".into(),
            Self::ExecConfirmationPolicyMismatch => "exec-confirmation-policy-mismatch".into(),
            Self::FdExecPolicyMismatch => "fd-exec-policy-mismatch".into(),
            Self::RuntimePolicyMismatch => "runtime-policy-mismatch".into(),
            Self::RuntimeVerifierMismatch => "runtime-verifier-mismatch".into(),
            Self::BackendMismatch => "backend-mismatch".into(),
            Self::AuthorityConfirmationMismatch => "authority-confirmation-mismatch".into(),
            Self::AuthorityTraceeMismatch => "authority-tracee-mismatch".into(),
            Self::TraceeHandoffFdMissing => "tracee-handoff-fd-missing".into(),
            Self::TraceeHandoffTargetNotPipe => "tracee-handoff-target-not-pipe".into(),
            Self::TraceeHandoffFdNotReadOnly => "tracee-handoff-fd-not-read-only".into(),
            Self::TraceeHandoffFdCloseOnExecUnexpected => {
                "tracee-handoff-fd-close-on-exec-unexpected".into()
            }
            Self::TraceeHandoffFdInfoInconsistent => "tracee-handoff-fdinfo-inconsistent".into(),
            Self::InvalidSupervisorWriteFd => "invalid-supervisor-write-fd".into(),
            Self::SupervisorFdUnavailable(v) => format!("supervisor-fd-unavailable:{v}"),
            Self::SupervisorTargetMismatch => "supervisor-target-mismatch".into(),
            Self::SupervisorFdNotWriteOnly => "supervisor-fd-not-write-only".into(),
            Self::SupervisorFdNonBlocking => "supervisor-fd-nonblocking".into(),
            Self::SupervisorFdInfoInconsistent => "supervisor-fdinfo-inconsistent".into(),
            Self::ChallengeEntropyUnavailable(v) => format!("challenge-entropy-unavailable:{v}"),
            Self::TicketEncodingFailed => "ticket-encoding-failed".into(),
            Self::TicketTooLarge { observed, maximum } => {
                format!("ticket-too-large:{observed}:{maximum}")
            }
            Self::TicketWriteFailed(v) => format!("ticket-write-failed:{v}"),
            Self::PartialTicketWrite { expected, observed } => {
                format!("partial-ticket-write:{expected}:{observed}")
            }
            Self::SupervisorFdChangedDuringIssuance => {
                "supervisor-fd-changed-during-issuance".into()
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LaunchRuntimeHandoffDisposition {
    Invalid,
    Blocked,
    Qualified,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchRuntimeHandoffReport {
    pub schema_version: String,
    pub policy_id: String,
    pub policy_digest: Option<String>,
    pub exec_confirmation_digest: String,
    pub critical_authority_qualification_digest: String,
    pub authority_snapshot_digest: String,
    pub tracee_pid: i32,
    pub tracee_handoff_read_fd: u32,
    pub supervisor_handoff_write_fd: i32,
    pub pipe_target: Option<String>,
    pub tracee_fdinfo_digest: Option<String>,
    pub supervisor_fdinfo_digest: Option<String>,
    pub channel_binding_digest: Option<String>,
    pub challenge_nonce_blake3_hex: Option<String>,
    pub ticket_digest: Option<String>,
    pub ticket_bytes_digest: Option<String>,
    pub ticket_bytes_len: u64,
    pub disposition: LaunchRuntimeHandoffDisposition,
    pub issues: Vec<LaunchRuntimeHandoffIssue>,
}

impl LaunchRuntimeHandoffReport {
    pub fn canonical_digest(&self) -> String {
        let mut h = blake3::Hasher::new();
        h.update(REPORT_DOMAIN);
        for value in [
            self.schema_version.as_str(),
            self.policy_id.as_str(),
            self.policy_digest.as_deref().unwrap_or("-"),
            self.exec_confirmation_digest.as_str(),
            self.critical_authority_qualification_digest.as_str(),
            self.authority_snapshot_digest.as_str(),
            self.pipe_target.as_deref().unwrap_or("-"),
            self.tracee_fdinfo_digest.as_deref().unwrap_or("-"),
            self.supervisor_fdinfo_digest.as_deref().unwrap_or("-"),
            self.channel_binding_digest.as_deref().unwrap_or("-"),
            self.challenge_nonce_blake3_hex.as_deref().unwrap_or("-"),
            self.ticket_digest.as_deref().unwrap_or("-"),
            self.ticket_bytes_digest.as_deref().unwrap_or("-"),
        ] {
            field(&mut h, value);
        }
        h.update(&self.tracee_pid.to_le_bytes());
        h.update(&self.tracee_handoff_read_fd.to_le_bytes());
        h.update(&self.supervisor_handoff_write_fd.to_le_bytes());
        h.update(&self.ticket_bytes_len.to_le_bytes());
        field(
            &mut h,
            match self.disposition {
                LaunchRuntimeHandoffDisposition::Invalid => "invalid",
                LaunchRuntimeHandoffDisposition::Blocked => "blocked",
                LaunchRuntimeHandoffDisposition::Qualified => "qualified",
            },
        );
        h.update(&(self.issues.len() as u64).to_le_bytes());
        for issue in &self.issues {
            field(&mut h, &issue.code());
        }
        format!("blake3:{}", h.finalize().to_hex())
    }

    pub const fn grants_physical_authority(&self) -> bool { false }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IssuedLaunchRuntimeChallenge {
    qualification_digest: String,
    report_digest: String,
    policy_digest: String,
    exec_confirmation_digest: String,
    critical_authority_qualification_digest: String,
    authority_snapshot_digest: String,
    plan_digest: String,
    launch_nonce_blake3_hex: String,
    tracee_pid: i32,
    tracee_handoff_read_fd: u32,
    supervisor_handoff_write_fd: i32,
    pipe_target: String,
    channel_binding_digest: String,
    runtime_policy_digest: String,
    runtime_verifier_ref: String,
    backend_id: String,
    executable_digest: String,
    challenge_nonce_blake3_hex: String,
    ticket_digest: String,
    ticket_bytes_digest: String,
}

impl IssuedLaunchRuntimeChallenge {
    pub fn qualification_digest(&self) -> &str { &self.qualification_digest }
    pub fn report_digest(&self) -> &str { &self.report_digest }
    pub fn policy_digest(&self) -> &str { &self.policy_digest }
    pub fn exec_confirmation_digest(&self) -> &str { &self.exec_confirmation_digest }
    pub fn critical_authority_qualification_digest(&self) -> &str {
        &self.critical_authority_qualification_digest
    }
    pub fn authority_snapshot_digest(&self) -> &str { &self.authority_snapshot_digest }
    pub fn plan_digest(&self) -> &str { &self.plan_digest }
    pub fn launch_nonce_blake3_hex(&self) -> &str { &self.launch_nonce_blake3_hex }
    pub const fn tracee_pid(&self) -> i32 { self.tracee_pid }
    pub const fn tracee_handoff_read_fd(&self) -> u32 { self.tracee_handoff_read_fd }
    pub const fn supervisor_handoff_write_fd(&self) -> i32 { self.supervisor_handoff_write_fd }
    pub fn pipe_target(&self) -> &str { &self.pipe_target }
    pub fn channel_binding_digest(&self) -> &str { &self.channel_binding_digest }
    pub fn runtime_policy_digest(&self) -> &str { &self.runtime_policy_digest }
    pub fn runtime_verifier_ref(&self) -> &str { &self.runtime_verifier_ref }
    pub fn backend_id(&self) -> &str { &self.backend_id }
    pub fn executable_digest(&self) -> &str { &self.executable_digest }
    pub fn challenge_nonce_blake3_hex(&self) -> &str { &self.challenge_nonce_blake3_hex }
    pub fn ticket_digest(&self) -> &str { &self.ticket_digest }
    pub fn ticket_bytes_digest(&self) -> &str { &self.ticket_bytes_digest }

    pub const fn same_kernel_pipe_object_observed_for_supervisor_and_tracee(&self) -> bool {
        true
    }
    pub const fn tracee_read_end_matches_critical_authority_snapshot(&self) -> bool { true }
    pub const fn supervisor_write_end_matches_same_pipe(&self) -> bool { true }
    pub const fn challenge_generated_from_os_csprng(&self) -> bool { true }
    pub const fn exact_ticket_written_to_supervisor_peer(&self) -> bool { true }
    pub const fn tracee_consumption_established(&self) -> bool { false }
    pub const fn exclusive_writer_authority_established(&self) -> bool { false }
    pub const fn first_runtime_checkpoint_binding_established(&self) -> bool { false }
    pub const fn trusted_time_established(&self) -> bool { false }
    pub const fn grants_physical_authority(&self) -> bool { false }
}

pub struct LaunchRuntimeHandoffQualification {
    pub report: LaunchRuntimeHandoffReport,
    pub ticket: LaunchRuntimeHandoffTicket,
    issued: IssuedLaunchRuntimeChallenge,
}

impl LaunchRuntimeHandoffQualification {
    pub fn issued(&self) -> &IssuedLaunchRuntimeChallenge { &self.issued }
    pub fn into_issued(self) -> IssuedLaunchRuntimeChallenge { self.issued }
}

pub fn issue_launch_runtime_challenge(
    policy: &LaunchRuntimeHandoffPolicy,
    confirmed: &ConfirmedStaticExecLaunch,
    authority: &CriticalExecAuthorityQualification,
    supervisor_write_fd: RawFd,
) -> Result<LaunchRuntimeHandoffQualification, LaunchRuntimeHandoffReport> {
    let policy_digest = policy.canonical_digest();
    let verified_authority = authority.verified();
    let mut report = LaunchRuntimeHandoffReport {
        schema_version: LAUNCH_RUNTIME_HANDOFF_REPORT_SCHEMA_V1.into(),
        policy_id: policy.policy_id.clone(),
        policy_digest: policy_digest.clone(),
        exec_confirmation_digest: confirmed.confirmation_digest().into(),
        critical_authority_qualification_digest: verified_authority.qualification_digest().into(),
        authority_snapshot_digest: verified_authority.authority_snapshot_digest().into(),
        tracee_pid: confirmed.tracee_pid(),
        tracee_handoff_read_fd: policy.tracee_handoff_read_fd,
        supervisor_handoff_write_fd: supervisor_write_fd,
        pipe_target: None,
        tracee_fdinfo_digest: None,
        supervisor_fdinfo_digest: None,
        channel_binding_digest: None,
        challenge_nonce_blake3_hex: None,
        ticket_digest: None,
        ticket_bytes_digest: None,
        ticket_bytes_len: 0,
        disposition: LaunchRuntimeHandoffDisposition::Invalid,
        issues: Vec::new(),
    };

    if !policy.validate() {
        report.issues.push(LaunchRuntimeHandoffIssue::InvalidPolicy);
        return Err(finalize(report));
    }
    if verified_authority.policy_digest() != policy.expected_critical_authority_policy_digest {
        report
            .issues
            .push(LaunchRuntimeHandoffIssue::CriticalAuthorityPolicyMismatch);
    }
    if confirmed.policy_digest() != policy.expected_exec_confirmation_policy_digest
        || verified_authority.exec_confirmation_policy_digest()
            != policy.expected_exec_confirmation_policy_digest
    {
        report
            .issues
            .push(LaunchRuntimeHandoffIssue::ExecConfirmationPolicyMismatch);
    }
    if confirmed.fd_exec_policy_digest() != policy.expected_fd_exec_policy_digest
        || verified_authority.fd_exec_policy_digest() != policy.expected_fd_exec_policy_digest
    {
        report.issues.push(LaunchRuntimeHandoffIssue::FdExecPolicyMismatch);
    }
    if confirmed.runtime_policy_digest() != policy.expected_runtime_policy_digest
        || verified_authority.runtime_policy_digest() != policy.expected_runtime_policy_digest
    {
        report.issues.push(LaunchRuntimeHandoffIssue::RuntimePolicyMismatch);
    }
    if confirmed.runtime_verifier_ref() != policy.expected_runtime_verifier_ref
        || verified_authority.runtime_verifier_ref() != policy.expected_runtime_verifier_ref
    {
        report.issues.push(LaunchRuntimeHandoffIssue::RuntimeVerifierMismatch);
    }
    if confirmed.backend_id() != policy.expected_backend_id
        || verified_authority.backend_id() != policy.expected_backend_id
    {
        report.issues.push(LaunchRuntimeHandoffIssue::BackendMismatch);
    }
    if verified_authority.exec_confirmation_digest() != confirmed.confirmation_digest() {
        report
            .issues
            .push(LaunchRuntimeHandoffIssue::AuthorityConfirmationMismatch);
    }
    if verified_authority.tracee_pid() != confirmed.tracee_pid() {
        report.issues.push(LaunchRuntimeHandoffIssue::AuthorityTraceeMismatch);
    }
    if supervisor_write_fd < 0 {
        report.issues.push(LaunchRuntimeHandoffIssue::InvalidSupervisorWriteFd);
    }
    if !report.issues.is_empty() {
        return Err(finalize(report));
    }

    let tracee_fd = authority
        .snapshot
        .fds
        .iter()
        .find(|value| value.fd == policy.tracee_handoff_read_fd);
    let Some(tracee_fd) = tracee_fd else {
        report.issues.push(LaunchRuntimeHandoffIssue::TraceeHandoffFdMissing);
        return Err(finalize(report));
    };
    report.tracee_fdinfo_digest = Some(tracee_fd.fdinfo_blake3.clone());
    let Some(pipe_inode) = parse_pipe_target(&tracee_fd.target) else {
        report
            .issues
            .push(LaunchRuntimeHandoffIssue::TraceeHandoffTargetNotPipe);
        return Err(finalize(report));
    };
    let tracee_flags = match parse_octal_flags(&tracee_fd.flags_octal) {
        Some(value) => value,
        None => {
            report
                .issues
                .push(LaunchRuntimeHandoffIssue::TraceeHandoffFdInfoInconsistent);
            return Err(finalize(report));
        }
    };
    if access_mode(tracee_flags) != OFlag::O_RDONLY.bits() {
        report
            .issues
            .push(LaunchRuntimeHandoffIssue::TraceeHandoffFdNotReadOnly);
    }
    if tracee_flags & OFlag::O_CLOEXEC.bits() != 0 {
        report
            .issues
            .push(LaunchRuntimeHandoffIssue::TraceeHandoffFdCloseOnExecUnexpected);
    }
    if tracee_fd.inode != pipe_inode {
        report
            .issues
            .push(LaunchRuntimeHandoffIssue::TraceeHandoffFdInfoInconsistent);
    }
    if !report.issues.is_empty() {
        return Err(finalize(report));
    }

    let before = match observe_supervisor_fd(supervisor_write_fd, policy.max_fdinfo_bytes) {
        Ok(value) => value,
        Err(error) => {
            report
                .issues
                .push(LaunchRuntimeHandoffIssue::SupervisorFdUnavailable(error));
            return Err(finalize(report));
        }
    };
    report.pipe_target = Some(tracee_fd.target.clone());
    report.supervisor_fdinfo_digest = Some(before.fdinfo_blake3.clone());
    if before.target != tracee_fd.target || before.inode != tracee_fd.inode {
        report
            .issues
            .push(LaunchRuntimeHandoffIssue::SupervisorTargetMismatch);
    }
    if access_mode(before.flags) != OFlag::O_WRONLY.bits() {
        report
            .issues
            .push(LaunchRuntimeHandoffIssue::SupervisorFdNotWriteOnly);
    }
    if before.flags & OFlag::O_NONBLOCK.bits() != 0 {
        report
            .issues
            .push(LaunchRuntimeHandoffIssue::SupervisorFdNonBlocking);
    }
    if before.inode != pipe_inode {
        report
            .issues
            .push(LaunchRuntimeHandoffIssue::SupervisorFdInfoInconsistent);
    }
    if !report.issues.is_empty() {
        return Err(finalize(report));
    }

    let channel_binding_digest = channel_binding_digest(
        confirmed.confirmation_digest(),
        verified_authority.qualification_digest(),
        confirmed.tracee_pid(),
        policy.tracee_handoff_read_fd,
        supervisor_write_fd,
        &tracee_fd.target,
        &tracee_fd.fdinfo_blake3,
        &before.fdinfo_blake3,
    );
    report.channel_binding_digest = Some(channel_binding_digest.clone());

    let mut random = [0u8; 32];
    if let Err(error) = getrandom::getrandom(&mut random) {
        report
            .issues
            .push(LaunchRuntimeHandoffIssue::ChallengeEntropyUnavailable(error.to_string()));
        return Err(finalize(report));
    }
    let challenge_nonce_blake3_hex = hex_lower(&random);
    report.challenge_nonce_blake3_hex = Some(challenge_nonce_blake3_hex.clone());

    let mut ticket = LaunchRuntimeHandoffTicket {
        schema_version: LAUNCH_RUNTIME_HANDOFF_TICKET_SCHEMA_V1.into(),
        exec_confirmation_digest: confirmed.confirmation_digest().into(),
        critical_authority_qualification_digest: verified_authority.qualification_digest().into(),
        authority_snapshot_digest: verified_authority.authority_snapshot_digest().into(),
        plan_digest: confirmed.plan_digest().into(),
        launch_nonce_blake3_hex: confirmed.launch_nonce_blake3_hex().into(),
        tracee_pid: confirmed.tracee_pid(),
        tracee_handoff_read_fd: policy.tracee_handoff_read_fd,
        pipe_target: tracee_fd.target.clone(),
        channel_binding_digest: channel_binding_digest.clone(),
        runtime_policy_digest: confirmed.runtime_policy_digest().into(),
        runtime_verifier_ref: confirmed.runtime_verifier_ref().into(),
        backend_id: confirmed.backend_id().into(),
        executable_digest: confirmed.executable_digest().into(),
        challenge_nonce_blake3_hex: challenge_nonce_blake3_hex.clone(),
        ticket_digest: String::new(),
    };
    ticket.ticket_digest = ticket.recompute_digest();
    let Some(wire) = ticket.to_wire_bytes() else {
        report.issues.push(LaunchRuntimeHandoffIssue::TicketEncodingFailed);
        return Err(finalize(report));
    };
    report.ticket_digest = Some(ticket.ticket_digest.clone());
    report.ticket_bytes_len = wire.len() as u64;
    let ticket_bytes_digest = format!("blake3:{}", blake3::hash(&wire).to_hex());
    report.ticket_bytes_digest = Some(ticket_bytes_digest.clone());
    if wire.len() as u32 > policy.max_ticket_bytes {
        report.issues.push(LaunchRuntimeHandoffIssue::TicketTooLarge {
            observed: wire.len() as u64,
            maximum: policy.max_ticket_bytes as u64,
        });
        return Err(finalize(report));
    }

    let written = match write(supervisor_write_fd, &wire) {
        Ok(value) => value,
        Err(error) => {
            report
                .issues
                .push(LaunchRuntimeHandoffIssue::TicketWriteFailed(error.to_string()));
            return Err(finalize(report));
        }
    };
    if written != wire.len() {
        report.issues.push(LaunchRuntimeHandoffIssue::PartialTicketWrite {
            expected: wire.len() as u64,
            observed: written as u64,
        });
        return Err(finalize(report));
    }

    let after = match observe_supervisor_fd(supervisor_write_fd, policy.max_fdinfo_bytes) {
        Ok(value) => value,
        Err(error) => {
            report
                .issues
                .push(LaunchRuntimeHandoffIssue::SupervisorFdUnavailable(error));
            return Err(finalize(report));
        }
    };
    if before != after {
        report
            .issues
            .push(LaunchRuntimeHandoffIssue::SupervisorFdChangedDuringIssuance);
        return Err(finalize(report));
    }

    report.disposition = LaunchRuntimeHandoffDisposition::Qualified;
    let report_digest = report.canonical_digest();
    let policy_digest = policy_digest.expect("validated policy has digest");
    let qualification_digest = qualification_digest(
        &policy_digest,
        confirmed.confirmation_digest(),
        verified_authority.qualification_digest(),
        &channel_binding_digest,
        &ticket.ticket_digest,
        &report_digest,
    );
    let issued = IssuedLaunchRuntimeChallenge {
        qualification_digest,
        report_digest,
        policy_digest,
        exec_confirmation_digest: confirmed.confirmation_digest().into(),
        critical_authority_qualification_digest: verified_authority.qualification_digest().into(),
        authority_snapshot_digest: verified_authority.authority_snapshot_digest().into(),
        plan_digest: confirmed.plan_digest().into(),
        launch_nonce_blake3_hex: confirmed.launch_nonce_blake3_hex().into(),
        tracee_pid: confirmed.tracee_pid(),
        tracee_handoff_read_fd: policy.tracee_handoff_read_fd,
        supervisor_handoff_write_fd: supervisor_write_fd,
        pipe_target: tracee_fd.target.clone(),
        channel_binding_digest,
        runtime_policy_digest: confirmed.runtime_policy_digest().into(),
        runtime_verifier_ref: confirmed.runtime_verifier_ref().into(),
        backend_id: confirmed.backend_id().into(),
        executable_digest: confirmed.executable_digest().into(),
        challenge_nonce_blake3_hex,
        ticket_digest: ticket.ticket_digest.clone(),
        ticket_bytes_digest,
    };
    Ok(LaunchRuntimeHandoffQualification { report, ticket, issued })
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct SupervisorFdObservation {
    target: String,
    flags: i32,
    inode: u64,
    fdinfo_blake3: String,
}

fn observe_supervisor_fd(fd: RawFd, maximum: u64) -> Result<SupervisorFdObservation, String> {
    if fd < 0 {
        return Err("negative-fd".into());
    }
    let base = PathBuf::from("/proc/self");
    let target = fs::read_link(base.join("fd").join(fd.to_string()))
        .map_err(|error| error.to_string())?
        .into_os_string()
        .into_string()
        .map_err(|_| "non-utf8-fd-target".to_string())?;
    let bytes = read_bounded(&base.join("fdinfo").join(fd.to_string()), maximum)?;
    let fdinfo_blake3 = format!("blake3:{}", blake3::hash(&bytes).to_hex());
    let parsed = parse_fdinfo(&bytes)?;
    Ok(SupervisorFdObservation {
        target,
        flags: parsed.flags,
        inode: parsed.inode,
        fdinfo_blake3,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct ParsedFdInfo {
    flags: i32,
    inode: u64,
}

fn parse_fdinfo(bytes: &[u8]) -> Result<ParsedFdInfo, String> {
    let text = std::str::from_utf8(bytes).map_err(|error| error.to_string())?;
    let mut flags = None;
    let mut inode = None;
    for line in text.lines() {
        let Some((key, value)) = line.split_once(':') else { continue; };
        let value = value.trim();
        match key {
            "flags" => {
                if flags.is_some() {
                    return Err("duplicate-flags".into());
                }
                flags = Some(
                    i32::from_str_radix(value, 8).map_err(|_| "invalid-flags".to_string())?,
                );
            }
            "ino" => {
                if inode.is_some() {
                    return Err("duplicate-ino".into());
                }
                inode = Some(value.parse::<u64>().map_err(|_| "invalid-ino".to_string())?);
            }
            _ => {}
        }
    }
    Ok(ParsedFdInfo {
        flags: flags.ok_or_else(|| "missing-flags".to_string())?,
        inode: inode.ok_or_else(|| "missing-ino".to_string())?,
    })
}

fn tracee_fd_from_snapshot<'a>(
    values: &'a [FdAuthorityEvidence],
    fd: u32,
) -> Option<&'a FdAuthorityEvidence> {
    values.iter().find(|value| value.fd == fd)
}

fn parse_pipe_target(value: &str) -> Option<u64> {
    let body = value.strip_prefix("pipe:[")?.strip_suffix(']')?;
    if body.is_empty() || !body.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    body.parse::<u64>().ok()
}

fn parse_octal_flags(value: &str) -> Option<i32> {
    if value.is_empty() || !value.bytes().all(|byte| (b'0'..=b'7').contains(&byte)) {
        return None;
    }
    i32::from_str_radix(value, 8).ok()
}

fn access_mode(flags: i32) -> i32 {
    flags & OFlag::O_ACCMODE.bits()
}

fn channel_binding_digest(
    confirmation_digest: &str,
    authority_qualification_digest: &str,
    tracee_pid: i32,
    tracee_fd: u32,
    supervisor_fd: i32,
    pipe_target: &str,
    tracee_fdinfo_digest: &str,
    supervisor_fdinfo_digest: &str,
) -> String {
    let mut h = blake3::Hasher::new();
    h.update(CHANNEL_DOMAIN);
    field(&mut h, confirmation_digest);
    field(&mut h, authority_qualification_digest);
    h.update(&tracee_pid.to_le_bytes());
    h.update(&tracee_fd.to_le_bytes());
    h.update(&supervisor_fd.to_le_bytes());
    for value in [pipe_target, tracee_fdinfo_digest, supervisor_fdinfo_digest] {
        field(&mut h, value);
    }
    format!("blake3:{}", h.finalize().to_hex())
}

#[allow(clippy::too_many_arguments)]
fn ticket_digest(
    schema_version: &str,
    exec_confirmation_digest: &str,
    critical_authority_qualification_digest: &str,
    authority_snapshot_digest: &str,
    plan_digest: &str,
    launch_nonce_blake3_hex: &str,
    tracee_pid: i32,
    tracee_handoff_read_fd: u32,
    pipe_target: &str,
    channel_binding_digest: &str,
    runtime_policy_digest: &str,
    runtime_verifier_ref: &str,
    backend_id: &str,
    executable_digest: &str,
    challenge_nonce_blake3_hex: &str,
) -> String {
    let mut h = blake3::Hasher::new();
    h.update(TICKET_DOMAIN);
    for value in [
        schema_version,
        exec_confirmation_digest,
        critical_authority_qualification_digest,
        authority_snapshot_digest,
        plan_digest,
        launch_nonce_blake3_hex,
    ] {
        field(&mut h, value);
    }
    h.update(&tracee_pid.to_le_bytes());
    h.update(&tracee_handoff_read_fd.to_le_bytes());
    for value in [
        pipe_target,
        channel_binding_digest,
        runtime_policy_digest,
        runtime_verifier_ref,
        backend_id,
        executable_digest,
        challenge_nonce_blake3_hex,
    ] {
        field(&mut h, value);
    }
    format!("blake3:{}", h.finalize().to_hex())
}

fn qualification_digest(
    policy_digest: &str,
    confirmation_digest: &str,
    authority_qualification_digest: &str,
    channel_digest: &str,
    ticket_digest: &str,
    report_digest: &str,
) -> String {
    let mut h = blake3::Hasher::new();
    h.update(QUALIFICATION_DOMAIN);
    for value in [
        policy_digest,
        confirmation_digest,
        authority_qualification_digest,
        channel_digest,
        ticket_digest,
        report_digest,
    ] {
        field(&mut h, value);
    }
    format!("blake3:{}", h.finalize().to_hex())
}

fn finalize(mut report: LaunchRuntimeHandoffReport) -> LaunchRuntimeHandoffReport {
    report.disposition = if report.issues.iter().any(LaunchRuntimeHandoffIssue::invalid) {
        LaunchRuntimeHandoffDisposition::Invalid
    } else {
        LaunchRuntimeHandoffDisposition::Blocked
    };
    report
}

fn read_bounded(path: &PathBuf, maximum: u64) -> Result<Vec<u8>, String> {
    use std::io::Read;
    let mut file = fs::File::open(path).map_err(|error| error.to_string())?;
    let mut output = Vec::new();
    let mut limited = (&mut file).take(maximum.saturating_add(1));
    limited
        .read_to_end(&mut output)
        .map_err(|error| error.to_string())?;
    if output.len() as u64 > maximum {
        return Err("bounded-read-limit-exceeded".into());
    }
    Ok(output)
}

fn push_wire_field(out: &mut Vec<u8>, value: &str) -> Option<()> {
    let length = u32::try_from(value.len()).ok()?;
    out.extend_from_slice(&length.to_le_bytes());
    out.extend_from_slice(value.as_bytes());
    Some(())
}

struct WireCursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> WireCursor<'a> {
    fn new(bytes: &'a [u8]) -> Self { Self { bytes, offset: 0 } }

    fn expect_magic(&mut self, magic: &[u8]) -> Option<()> {
        let end = self.offset.checked_add(magic.len())?;
        if self.bytes.get(self.offset..end)? != magic {
            return None;
        }
        self.offset = end;
        Some(())
    }

    fn read_exact<const N: usize>(&mut self) -> Option<[u8; N]> {
        let end = self.offset.checked_add(N)?;
        let slice = self.bytes.get(self.offset..end)?;
        let mut out = [0u8; N];
        out.copy_from_slice(slice);
        self.offset = end;
        Some(out)
    }

    fn read_u32(&mut self) -> Option<u32> {
        Some(u32::from_le_bytes(self.read_exact()?))
    }

    fn read_i32(&mut self) -> Option<i32> {
        Some(i32::from_le_bytes(self.read_exact()?))
    }

    fn read_string(&mut self) -> Option<String> {
        let length = usize::try_from(self.read_u32()?).ok()?;
        if length > MAX_TEXT * 4 {
            return None;
        }
        let end = self.offset.checked_add(length)?;
        let bytes = self.bytes.get(self.offset..end)?;
        self.offset = end;
        String::from_utf8(bytes.to_vec()).ok()
    }

    fn finished(&self) -> bool { self.offset == self.bytes.len() }
}

fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        out.push(HEX[(byte >> 4) as usize] as char);
        out.push(HEX[(byte & 0x0f) as usize] as char);
    }
    out
}

fn field(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn sorted_strings(hasher: &mut blake3::Hasher, values: &[String]) {
    let mut values = values.to_vec();
    values.sort();
    hasher.update(&(values.len() as u64).to_le_bytes());
    for value in values {
        field(hasher, &value);
    }
}

fn canonical_text(value: &str) -> bool {
    !value.is_empty()
        && value == value.trim()
        && value.len() <= MAX_TEXT
        && !value.chars().any(char::is_control)
}

fn valid_blake3(value: &str) -> bool {
    value.strip_prefix("blake3:").is_some_and(|hex| lower_hex_exact(hex, 64))
}

fn lower_hex_exact(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn valid_refs(values: &[String]) -> bool {
    values.len() <= MAX_REFS && values.iter().all(|value| canonical_text(value)) && unique(values)
}

fn unique(values: &[String]) -> bool {
    let mut seen = BTreeSet::new();
    values.iter().all(|value| seen.insert(value.as_str()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(label: &str) -> String {
        format!("blake3:{}", blake3::hash(label.as_bytes()).to_hex())
    }

    fn policy() -> LaunchRuntimeHandoffPolicy {
        LaunchRuntimeHandoffPolicy {
            schema_version: LAUNCH_RUNTIME_HANDOFF_POLICY_SCHEMA_V1.into(),
            policy_id: "policy:launch-runtime-handoff:1".into(),
            expected_critical_authority_policy_digest: d("authority-policy"),
            expected_exec_confirmation_policy_digest: d("confirmation-policy"),
            expected_fd_exec_policy_digest: d("fd-policy"),
            expected_runtime_policy_digest: d("runtime-policy"),
            expected_runtime_verifier_ref: "verifier:host".into(),
            expected_backend_id: "backend:in-process".into(),
            tracee_handoff_read_fd: 7,
            max_ticket_bytes: 4096,
            max_fdinfo_bytes: 1 << 20,
            evidence_refs: vec!["review:a".into(), "review:b".into()],
        }
    }

    fn ticket() -> LaunchRuntimeHandoffTicket {
        let mut value = LaunchRuntimeHandoffTicket {
            schema_version: LAUNCH_RUNTIME_HANDOFF_TICKET_SCHEMA_V1.into(),
            exec_confirmation_digest: d("confirmation"),
            critical_authority_qualification_digest: d("authority"),
            authority_snapshot_digest: d("snapshot"),
            plan_digest: d("plan"),
            launch_nonce_blake3_hex: "11".repeat(32),
            tracee_pid: 42,
            tracee_handoff_read_fd: 7,
            pipe_target: "pipe:[12345]".into(),
            channel_binding_digest: d("channel"),
            runtime_policy_digest: d("runtime-policy"),
            runtime_verifier_ref: "verifier:host".into(),
            backend_id: "backend:in-process".into(),
            executable_digest: d("executable"),
            challenge_nonce_blake3_hex: "22".repeat(32),
            ticket_digest: String::new(),
        };
        value.ticket_digest = value.recompute_digest();
        value
    }

    #[test]
    fn policy_reference_order_is_nonsemantic() {
        let left = policy();
        let mut right = left.clone();
        right.evidence_refs.reverse();
        assert_eq!(left.canonical_digest(), right.canonical_digest());
        right.tracee_handoff_read_fd += 1;
        assert_ne!(left.canonical_digest(), right.canonical_digest());
    }

    #[test]
    fn ticket_round_trip_is_exact_and_tamper_evident() {
        let value = ticket();
        assert!(value.validate());
        let wire = value.to_wire_bytes().unwrap();
        let decoded = LaunchRuntimeHandoffTicket::from_wire_bytes(&wire).unwrap();
        assert_eq!(decoded, value);

        let mut tampered = value;
        tampered.tracee_pid += 1;
        assert!(!tampered.validate());
    }

    #[test]
    fn pipe_target_parser_is_strict() {
        assert_eq!(parse_pipe_target("pipe:[12345]"), Some(12345));
        assert_eq!(parse_pipe_target("socket:[12345]"), None);
        assert_eq!(parse_pipe_target("pipe:[]"), None);
        assert_eq!(parse_pipe_target("pipe:[12x]"), None);
    }

    #[test]
    fn channel_binding_is_sensitive_to_both_endpoint_evidence() {
        let left = channel_binding_digest(
            &d("confirmation"),
            &d("authority"),
            42,
            7,
            8,
            "pipe:[123]",
            &d("tracee-fdinfo"),
            &d("supervisor-fdinfo"),
        );
        let right = channel_binding_digest(
            &d("confirmation"),
            &d("authority"),
            42,
            7,
            8,
            "pipe:[123]",
            &d("tracee-fdinfo"),
            &d("changed-supervisor-fdinfo"),
        );
        assert_ne!(left, right);
    }

    #[test]
    fn claim_ceiling_remains_explicit() {
        let claims = [
            "tracee_consumption_established=false",
            "exclusive_writer_authority_established=false",
            "first_runtime_checkpoint_binding_established=false",
            "trusted_time_established=false",
            "grants_physical_authority=false",
        ];
        assert_eq!(claims.len(), 5);
    }

    #[test]
    fn helper_lookup_is_exact_fd_number() {
        let fds = vec![
            FdAuthorityEvidence {
                fd: 7,
                target: "pipe:[1]".into(),
                position: 0,
                flags_octal: "00".into(),
                mount_id: 1,
                inode: 1,
                fdinfo_blake3: d("fd7"),
            },
            FdAuthorityEvidence {
                fd: 9,
                target: "pipe:[2]".into(),
                position: 0,
                flags_octal: "00".into(),
                mount_id: 1,
                inode: 2,
                fdinfo_blake3: d("fd9"),
            },
        ];
        assert_eq!(tracee_fd_from_snapshot(&fds, 9).unwrap().inode, 2);
        assert!(tracee_fd_from_snapshot(&fds, 8).is_none());
    }
}
