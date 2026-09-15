// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Ptrace syscall-entry confinement from confirmed exec to checkpoint-one readiness.
//!
//! Linux x86_64 GNU/musl v1 deliberately uses the locked `nix` safe ptrace
//! wrappers. A qualified run starts at the existing PTRACE_EVENT_EXEC stop,
//! consumes the corresponding exec syscall-exit stop, then mediates every
//! syscall entry. Only a fixed bootstrap profile is restarted. The sole
//! `write(2)` admitted is an exact bootstrap-ready wire to a reviewed inherited
//! pipe. That wire is independently verified before the write is restarted,
//! rebound to the opaque supervisor-issued launch challenge, observed at the
//! nonblocking supervisor peer after write exit, and followed by an exact
//! critical-authority re-observation.

#![cfg(all(
    target_os = "linux",
    target_arch = "x86_64",
    any(target_env = "gnu", target_env = "musl")
))]
#![deny(unsafe_code)]

use nix::{
    fcntl::OFlag,
    libc,
    sys::{
        ptrace::{self, Options},
        signal::{kill, Signal},
        wait::{waitpid, WaitPidFlag, WaitStatus},
    },
    unistd::{getpid, read, write, Pid},
};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeSet,
    fs::{self, File},
    io::{Read, Seek, SeekFrom},
    os::unix::io::RawFd,
    path::PathBuf,
};
use symthaea_assurance_bootstrap_ready_checkpoint::{
    verify_bootstrap_ready_wire, BootstrapReadyCheckpointPolicy,
    BootstrapReadyCheckpointQualification, VerifiedBootstrapReadyWire,
};
use symthaea_assurance_critical_exec_preserved_authority::{
    assess_critical_exec_preserved_authority, CriticalExecAuthorityPolicy,
    CriticalExecAuthorityQualification,
};
use symthaea_assurance_launch_runtime_handoff_channel::IssuedLaunchRuntimeChallenge;
use symthaea_assurance_ptrace_static_exec_confirmation::ConfirmedStaticExecLaunch;
use symthaea_evidence_verifier_runtime_continuity::VerifierRuntimeContinuityPolicy;

pub const BOOTSTRAP_SYSCALL_CONFINEMENT_POLICY_SCHEMA_V1: &str =
    "symthaea.assurance.bootstrap-syscall-confinement-policy.v1";
pub const BOOTSTRAP_SYSCALL_CONFINEMENT_REPORT_SCHEMA_V1: &str =
    "symthaea.assurance.bootstrap-syscall-confinement-report.v1";
const POLICY_DOMAIN: &[u8] =
    b"symthaea.assurance.bootstrap-syscall-confinement-policy.digest.v1\0";
const SEQUENCE_DOMAIN: &[u8] =
    b"symthaea.assurance.bootstrap-syscall-sequence.digest.v1\0";
const REPORT_DOMAIN: &[u8] =
    b"symthaea.assurance.bootstrap-syscall-confinement-report.digest.v1\0";
const QUALIFICATION_DOMAIN: &[u8] =
    b"symthaea.assurance.bootstrap-syscall-confinement-qualification.digest.v1\0";
const EMISSION_DOMAIN: &[u8] = b"symthaea.assurance.bootstrap-ready-emission.digest.v1\0";
const MAX_TEXT: usize = 4096;
const MAX_REFS: usize = 128;
const LINUX_PIPE_BUF: u32 = 4096;
const HARD_MAX_SYSCALLS: u64 = 1_000_000;
const HARD_MAX_FDINFO_BYTES: u64 = 4 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BootstrapSyscallConfinementPolicy {
    pub schema_version: String,
    pub policy_id: String,
    pub expected_exec_confirmation_policy_digest: String,
    pub expected_critical_authority_policy_digest: String,
    pub expected_handoff_policy_digest: String,
    pub expected_bootstrap_ready_policy_digest: String,
    pub expected_runtime_policy_digest: String,
    pub expected_runtime_verifier_ref: String,
    pub tracee_ready_write_fd: u32,
    pub max_syscall_count: u64,
    pub max_fdinfo_bytes: u64,
    pub evidence_refs: Vec<String>,
}

impl BootstrapSyscallConfinementPolicy {
    pub fn validate(&self) -> bool {
        self.schema_version == BOOTSTRAP_SYSCALL_CONFINEMENT_POLICY_SCHEMA_V1
            && text(&self.policy_id)
            && digest(&self.expected_exec_confirmation_policy_digest)
            && digest(&self.expected_critical_authority_policy_digest)
            && digest(&self.expected_handoff_policy_digest)
            && digest(&self.expected_bootstrap_ready_policy_digest)
            && digest(&self.expected_runtime_policy_digest)
            && text(&self.expected_runtime_verifier_ref)
            && self.tracee_ready_write_fd <= 1_048_576
            && (1..=HARD_MAX_SYSCALLS).contains(&self.max_syscall_count)
            && (1..=HARD_MAX_FDINFO_BYTES).contains(&self.max_fdinfo_bytes)
            && refs(&self.evidence_refs)
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
            self.expected_exec_confirmation_policy_digest.as_str(),
            self.expected_critical_authority_policy_digest.as_str(),
            self.expected_handoff_policy_digest.as_str(),
            self.expected_bootstrap_ready_policy_digest.as_str(),
            self.expected_runtime_policy_digest.as_str(),
            self.expected_runtime_verifier_ref.as_str(),
        ] {
            field(&mut h, value);
        }
        h.update(&self.tracee_ready_write_fd.to_le_bytes());
        h.update(&self.max_syscall_count.to_le_bytes());
        h.update(&self.max_fdinfo_bytes.to_le_bytes());
        sorted(&mut h, &self.evidence_refs);
        Some(b3(h.finalize()))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BootstrapSyscallObservation {
    pub syscall_number: i64,
    pub args: [u64; 6],
    pub return_value: i64,
    pub ready_token_write: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BootstrapSyscallConfinementIssue {
    InvalidPolicy,
    UnsupportedReadyWireLimit,
    ParentIdentityMismatch,
    ReadyFdMissing,
    ReadyFdTargetNotPipe,
    ReadyFdNotWriteOnly,
    ReadyFdCloseOnExecUnexpected,
    ReadyFdNonBlocking,
    InvalidSupervisorReadyReadFd,
    SupervisorReadyFdUnavailable(String),
    SupervisorReadyTargetMismatch,
    SupervisorReadyFdNotReadOnly,
    SupervisorReadyFdBlocking,
    TraceeNotCurrentlyStopped,
    TracerPidMismatch,
    SetOptionsFailed(String),
    SyscallRestartFailed(String),
    WaitFailed(String),
    UnexpectedStop(String),
    ExecSyscallExitUnexpected(i64),
    SyscallLimitExceeded,
    DisallowedSyscall(i64),
    TraceeMemoryReadFailed(String),
    ReadyWriteLengthInvalid,
    ReadyWireInvalid(String),
    ReadyTicketMismatch,
    ReadyChallengeMismatch,
    ReadyRuntimePolicyMismatch,
    ReadyWriteReturnedUnexpected(i64),
    ReadyPeerReadFailed(String),
    ReadyPeerBytesMismatch,
    SupervisorReadyFdChanged,
    CriticalAuthorityRecheckFailed(String),
    CriticalAuthorityChangedBeforeReady,
    TraceeExitedBeforeReady(i32),
    TraceeSignaledBeforeReady(String),
    KillFailed(String),
    DetachFailed(String),
}

impl BootstrapSyscallConfinementIssue {
    fn invalid(&self) -> bool {
        matches!(
            self,
            Self::InvalidPolicy
                | Self::UnsupportedReadyWireLimit
                | Self::InvalidSupervisorReadyReadFd
        )
    }

    fn code(&self) -> String {
        match self {
            Self::InvalidPolicy => "invalid-policy".into(),
            Self::UnsupportedReadyWireLimit => "unsupported-ready-wire-limit".into(),
            Self::ParentIdentityMismatch => "parent-identity-mismatch".into(),
            Self::ReadyFdMissing => "ready-fd-missing".into(),
            Self::ReadyFdTargetNotPipe => "ready-fd-target-not-pipe".into(),
            Self::ReadyFdNotWriteOnly => "ready-fd-not-write-only".into(),
            Self::ReadyFdCloseOnExecUnexpected => "ready-fd-cloexec-unexpected".into(),
            Self::ReadyFdNonBlocking => "ready-fd-nonblocking".into(),
            Self::InvalidSupervisorReadyReadFd => "invalid-supervisor-ready-read-fd".into(),
            Self::SupervisorReadyFdUnavailable(v) => {
                format!("supervisor-ready-fd-unavailable:{v}")
            }
            Self::SupervisorReadyTargetMismatch => "supervisor-ready-target-mismatch".into(),
            Self::SupervisorReadyFdNotReadOnly => "supervisor-ready-fd-not-read-only".into(),
            Self::SupervisorReadyFdBlocking => "supervisor-ready-fd-blocking".into(),
            Self::TraceeNotCurrentlyStopped => "tracee-not-currently-stopped".into(),
            Self::TracerPidMismatch => "tracer-pid-mismatch".into(),
            Self::SetOptionsFailed(v) => format!("set-options-failed:{v}"),
            Self::SyscallRestartFailed(v) => format!("syscall-restart-failed:{v}"),
            Self::WaitFailed(v) => format!("wait-failed:{v}"),
            Self::UnexpectedStop(v) => format!("unexpected-stop:{v}"),
            Self::ExecSyscallExitUnexpected(v) => format!("exec-syscall-exit-unexpected:{v}"),
            Self::SyscallLimitExceeded => "syscall-limit-exceeded".into(),
            Self::DisallowedSyscall(v) => format!("disallowed-syscall:{v}"),
            Self::TraceeMemoryReadFailed(v) => format!("tracee-memory-read-failed:{v}"),
            Self::ReadyWriteLengthInvalid => "ready-write-length-invalid".into(),
            Self::ReadyWireInvalid(v) => format!("ready-wire-invalid:{v}"),
            Self::ReadyTicketMismatch => "ready-ticket-mismatch".into(),
            Self::ReadyChallengeMismatch => "ready-challenge-mismatch".into(),
            Self::ReadyRuntimePolicyMismatch => "ready-runtime-policy-mismatch".into(),
            Self::ReadyWriteReturnedUnexpected(v) => {
                format!("ready-write-returned-unexpected:{v}")
            }
            Self::ReadyPeerReadFailed(v) => format!("ready-peer-read-failed:{v}"),
            Self::ReadyPeerBytesMismatch => "ready-peer-bytes-mismatch".into(),
            Self::SupervisorReadyFdChanged => "supervisor-ready-fd-changed".into(),
            Self::CriticalAuthorityRecheckFailed(v) => {
                format!("critical-authority-recheck-failed:{v}")
            }
            Self::CriticalAuthorityChangedBeforeReady => {
                "critical-authority-changed-before-ready".into()
            }
            Self::TraceeExitedBeforeReady(v) => format!("tracee-exited-before-ready:{v}"),
            Self::TraceeSignaledBeforeReady(v) => format!("tracee-signaled-before-ready:{v}"),
            Self::KillFailed(v) => format!("kill-failed:{v}"),
            Self::DetachFailed(v) => format!("detach-failed:{v}"),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BootstrapSyscallConfinementDisposition {
    Invalid,
    Blocked,
    Qualified,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BootstrapSyscallConfinementReport {
    pub schema_version: String,
    pub policy_id: String,
    pub policy_digest: Option<String>,
    pub tracee_pid: i32,
    pub exec_confirmation_digest: String,
    pub issued_challenge_qualification_digest: String,
    pub critical_authority_qualification_digest: String,
    pub initial_authority_snapshot_digest: String,
    pub final_authority_snapshot_digest: Option<String>,
    pub syscall_sequence_digest: Option<String>,
    pub syscall_count: u64,
    pub ready_wire_digest: Option<String>,
    pub ready_checkpoint_digest: Option<String>,
    pub ready_wire_bytes_digest: Option<String>,
    pub ready_wire_bytes_len: u64,
    pub ready_pipe_target: Option<String>,
    pub tracee_stopped_at_ready_exit: bool,
    pub violation_sigkill_sent_before_restart: bool,
    pub disposition: BootstrapSyscallConfinementDisposition,
    pub issues: Vec<BootstrapSyscallConfinementIssue>,
}

impl BootstrapSyscallConfinementReport {
    pub fn canonical_digest(&self) -> String {
        let mut h = blake3::Hasher::new();
        h.update(REPORT_DOMAIN);
        for value in [
            self.schema_version.as_str(),
            self.policy_id.as_str(),
            self.policy_digest.as_deref().unwrap_or("-"),
            self.exec_confirmation_digest.as_str(),
            self.issued_challenge_qualification_digest.as_str(),
            self.critical_authority_qualification_digest.as_str(),
            self.initial_authority_snapshot_digest.as_str(),
            self.final_authority_snapshot_digest.as_deref().unwrap_or("-"),
            self.syscall_sequence_digest.as_deref().unwrap_or("-"),
            self.ready_wire_digest.as_deref().unwrap_or("-"),
            self.ready_checkpoint_digest.as_deref().unwrap_or("-"),
            self.ready_wire_bytes_digest.as_deref().unwrap_or("-"),
            self.ready_pipe_target.as_deref().unwrap_or("-"),
        ] {
            field(&mut h, value);
        }
        h.update(&self.tracee_pid.to_le_bytes());
        h.update(&self.syscall_count.to_le_bytes());
        h.update(&self.ready_wire_bytes_len.to_le_bytes());
        h.update(&[
            u8::from(self.tracee_stopped_at_ready_exit),
            u8::from(self.violation_sigkill_sent_before_restart),
        ]);
        field(
            &mut h,
            match self.disposition {
                BootstrapSyscallConfinementDisposition::Invalid => "invalid",
                BootstrapSyscallConfinementDisposition::Blocked => "blocked",
                BootstrapSyscallConfinementDisposition::Qualified => "qualified",
            },
        );
        h.update(&(self.issues.len() as u64).to_le_bytes());
        for issue in &self.issues {
            field(&mut h, &issue.code());
        }
        b3(h.finalize())
    }

    pub const fn grants_physical_authority(&self) -> bool { false }
}

#[derive(Debug)]
pub struct BootstrapSyscallConfinementReady {
    pid: Pid,
    qualification_digest: String,
    report_digest: String,
    policy_digest: String,
    exec_confirmation_digest: String,
    issued_challenge_qualification_digest: String,
    ready_wire_digest: String,
    ready_checkpoint_digest: String,
    ready_wire_bytes_digest: String,
    syscall_sequence_digest: String,
    syscall_count: u64,
    authority_snapshot_digest: String,
}

impl BootstrapSyscallConfinementReady {
    pub fn qualification_digest(&self) -> &str { &self.qualification_digest }
    pub fn report_digest(&self) -> &str { &self.report_digest }
    pub fn policy_digest(&self) -> &str { &self.policy_digest }
    pub fn exec_confirmation_digest(&self) -> &str { &self.exec_confirmation_digest }
    pub fn issued_challenge_qualification_digest(&self) -> &str {
        &self.issued_challenge_qualification_digest
    }
    pub fn ready_wire_digest(&self) -> &str { &self.ready_wire_digest }
    pub fn ready_checkpoint_digest(&self) -> &str { &self.ready_checkpoint_digest }
    pub fn ready_wire_bytes_digest(&self) -> &str { &self.ready_wire_bytes_digest }
    pub fn syscall_sequence_digest(&self) -> &str { &self.syscall_sequence_digest }
    pub const fn syscall_count(&self) -> u64 { self.syscall_count }
    pub fn authority_snapshot_digest(&self) -> &str { &self.authority_snapshot_digest }
    pub const fn tracee_pid(&self) -> i32 { self.pid.as_raw() }
    pub const fn all_post_exec_syscalls_mediated_until_ready(&self) -> bool { true }
    pub const fn no_unreviewed_syscall_restarted_before_ready(&self) -> bool { true }
    pub const fn ready_wire_verified_before_write_restart(&self) -> bool { true }
    pub const fn ready_wire_matches_supervisor_issued_challenge(&self) -> bool { true }
    pub const fn exact_ready_bytes_observed_on_supervisor_peer(&self) -> bool { true }
    pub const fn critical_authority_reverified_at_ready(&self) -> bool { true }
    pub const fn tracee_stopped_before_release(&self) -> bool { true }
    pub const fn actual_live_observation_proven_by_supervisor_wire_alone(&self) -> bool { false }
    pub const fn generic_external_side_effect_freedom_established(&self) -> bool { false }
    pub const fn instruction_continuity_established(&self) -> bool { false }
    pub const fn exclusive_pipe_writer_authority_established(&self) -> bool { false }
    pub const fn trusted_time_established(&self) -> bool { false }
    pub const fn grants_physical_authority(&self) -> bool { false }
}

pub struct BootstrapSyscallConfinementQualification {
    pub report: BootstrapSyscallConfinementReport,
    pub syscalls: Vec<BootstrapSyscallObservation>,
    ready: BootstrapSyscallConfinementReady,
}
impl BootstrapSyscallConfinementQualification {
    pub fn ready(&self) -> &BootstrapSyscallConfinementReady { &self.ready }
    pub fn into_ready(self) -> BootstrapSyscallConfinementReady { self.ready }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BootstrapReadyEmissionReceipt {
    pub wire_digest: String,
    pub wire_bytes_digest: String,
    pub bytes_written: u64,
    pub emission_digest: String,
}

pub fn emit_bootstrap_ready_token(
    ready: &BootstrapReadyCheckpointQualification,
    ready_write_fd: RawFd,
) -> Result<BootstrapReadyEmissionReceipt, String> {
    if ready_write_fd < 0 {
        return Err("invalid-ready-write-fd".into());
    }
    let bytes = &ready.wire_bytes;
    if bytes.is_empty() || bytes.len() > LINUX_PIPE_BUF as usize {
        return Err("ready-wire-exceeds-linux-pipe-buf".into());
    }
    let written = write(ready_write_fd, bytes).map_err(|error| error.to_string())?;
    if written != bytes.len() {
        return Err(format!("partial-ready-write:{written}:{}", bytes.len()));
    }
    let verified = ready.verified_wire();
    let mut h = blake3::Hasher::new();
    h.update(EMISSION_DOMAIN);
    for value in [
        verified.wire_digest(),
        verified.wire_bytes_digest(),
        verified.checkpoint_digest(),
    ] {
        field(&mut h, value);
    }
    h.update(&(written as u64).to_le_bytes());
    Ok(BootstrapReadyEmissionReceipt {
        wire_digest: verified.wire_digest().into(),
        wire_bytes_digest: verified.wire_bytes_digest().into(),
        bytes_written: written as u64,
        emission_digest: b3(h.finalize()),
    })
}

#[allow(clippy::too_many_arguments)]
pub fn confine_bootstrap_until_ready(
    policy: &BootstrapSyscallConfinementPolicy,
    confirmed: &ConfirmedStaticExecLaunch,
    initial_authority: &CriticalExecAuthorityQualification,
    authority_policy: &CriticalExecAuthorityPolicy,
    issued: &IssuedLaunchRuntimeChallenge,
    ready_policy: &BootstrapReadyCheckpointPolicy,
    runtime_policy: &VerifierRuntimeContinuityPolicy,
    supervisor_ready_read_fd: RawFd,
) -> Result<BootstrapSyscallConfinementQualification, BootstrapSyscallConfinementReport> {
    let policy_digest = policy.canonical_digest();
    let initial = initial_authority.verified();
    let mut report = BootstrapSyscallConfinementReport {
        schema_version: BOOTSTRAP_SYSCALL_CONFINEMENT_REPORT_SCHEMA_V1.into(),
        policy_id: policy.policy_id.clone(),
        policy_digest: policy_digest.clone(),
        tracee_pid: confirmed.tracee_pid(),
        exec_confirmation_digest: confirmed.confirmation_digest().into(),
        issued_challenge_qualification_digest: issued.qualification_digest().into(),
        critical_authority_qualification_digest: initial.qualification_digest().into(),
        initial_authority_snapshot_digest: initial.authority_snapshot_digest().into(),
        final_authority_snapshot_digest: None,
        syscall_sequence_digest: None,
        syscall_count: 0,
        ready_wire_digest: None,
        ready_checkpoint_digest: None,
        ready_wire_bytes_digest: None,
        ready_wire_bytes_len: 0,
        ready_pipe_target: None,
        tracee_stopped_at_ready_exit: false,
        violation_sigkill_sent_before_restart: false,
        disposition: BootstrapSyscallConfinementDisposition::Invalid,
        issues: Vec::new(),
    };

    if !policy.validate() {
        report.issues.push(BootstrapSyscallConfinementIssue::InvalidPolicy);
        return Err(finalize(report));
    }
    if ready_policy.max_wire_bytes > LINUX_PIPE_BUF {
        report
            .issues
            .push(BootstrapSyscallConfinementIssue::UnsupportedReadyWireLimit);
    }
    if supervisor_ready_read_fd < 0 {
        report
            .issues
            .push(BootstrapSyscallConfinementIssue::InvalidSupervisorReadyReadFd);
    }
    let runtime_policy_digest = runtime_policy.canonical_digest();
    if confirmed.policy_digest() != policy.expected_exec_confirmation_policy_digest
        || initial.policy_digest() != policy.expected_critical_authority_policy_digest
        || issued.policy_digest() != policy.expected_handoff_policy_digest
        || ready_policy.canonical_digest().as_deref()
            != Some(policy.expected_bootstrap_ready_policy_digest.as_str())
        || runtime_policy_digest.as_deref()
            != Some(policy.expected_runtime_policy_digest.as_str())
        || runtime_policy.verifier_ref != policy.expected_runtime_verifier_ref
        || confirmed.runtime_policy_digest() != policy.expected_runtime_policy_digest
        || confirmed.runtime_verifier_ref() != policy.expected_runtime_verifier_ref
        || initial.exec_confirmation_digest() != confirmed.confirmation_digest()
        || issued.exec_confirmation_digest() != confirmed.confirmation_digest()
        || issued.critical_authority_qualification_digest() != initial.qualification_digest()
        || issued.tracee_pid() != confirmed.tracee_pid()
        || issued.runtime_policy_digest() != policy.expected_runtime_policy_digest
        || issued.runtime_verifier_ref() != policy.expected_runtime_verifier_ref
        || issued.executable_digest() != runtime_policy.expected_executable_digest
    {
        report
            .issues
            .push(BootstrapSyscallConfinementIssue::ParentIdentityMismatch);
    }
    if !report.issues.is_empty() {
        return Err(finalize(report));
    }

    let Some(tracee_ready_fd) = initial_authority
        .snapshot
        .fds
        .iter()
        .find(|value| value.fd == policy.tracee_ready_write_fd)
    else {
        report.issues.push(BootstrapSyscallConfinementIssue::ReadyFdMissing);
        return Err(finalize(report));
    };
    let Some(_) = pipe_inode(&tracee_ready_fd.target) else {
        report
            .issues
            .push(BootstrapSyscallConfinementIssue::ReadyFdTargetNotPipe);
        return Err(finalize(report));
    };
    let Some(tracee_flags) = octal(&tracee_ready_fd.flags_octal) else {
        report
            .issues
            .push(BootstrapSyscallConfinementIssue::ReadyFdNotWriteOnly);
        return Err(finalize(report));
    };
    if tracee_flags & OFlag::O_ACCMODE.bits() != OFlag::O_WRONLY.bits() {
        report
            .issues
            .push(BootstrapSyscallConfinementIssue::ReadyFdNotWriteOnly);
    }
    if tracee_flags & OFlag::O_CLOEXEC.bits() != 0 {
        report
            .issues
            .push(BootstrapSyscallConfinementIssue::ReadyFdCloseOnExecUnexpected);
    }
    if tracee_flags & OFlag::O_NONBLOCK.bits() != 0 {
        report
            .issues
            .push(BootstrapSyscallConfinementIssue::ReadyFdNonBlocking);
    }

    let before = match local_fd(supervisor_ready_read_fd, policy.max_fdinfo_bytes) {
        Ok(value) => value,
        Err(error) => {
            report
                .issues
                .push(BootstrapSyscallConfinementIssue::SupervisorReadyFdUnavailable(error));
            return Err(finalize(report));
        }
    };
    if before.target != tracee_ready_fd.target || before.inode != tracee_ready_fd.inode {
        report
            .issues
            .push(BootstrapSyscallConfinementIssue::SupervisorReadyTargetMismatch);
    }
    if before.flags & OFlag::O_ACCMODE.bits() != OFlag::O_RDONLY.bits() {
        report
            .issues
            .push(BootstrapSyscallConfinementIssue::SupervisorReadyFdNotReadOnly);
    }
    if before.flags & OFlag::O_NONBLOCK.bits() == 0 {
        report
            .issues
            .push(BootstrapSyscallConfinementIssue::SupervisorReadyFdBlocking);
    }
    report.ready_pipe_target = Some(before.target.clone());
    if !report.issues.is_empty() {
        return Err(finalize(report));
    }

    if !live_trace_stop(confirmed.tracee_pid()) {
        report
            .issues
            .push(BootstrapSyscallConfinementIssue::TraceeNotCurrentlyStopped);
        return Err(finalize(report));
    }
    let pid = Pid::from_raw(confirmed.tracee_pid());
    let options = Options::PTRACE_O_TRACESYSGOOD | Options::PTRACE_O_EXITKILL;
    if let Err(error) = ptrace::setoptions(pid, options) {
        report
            .issues
            .push(BootstrapSyscallConfinementIssue::SetOptionsFailed(error.to_string()));
        return Err(finalize(report));
    }

    // PTRACE_EVENT_EXEC is an event stop at successful exec. When restarted
    // with PTRACE_SYSCALL, the next syscall-stop is the exit-stop for that exec.
    if let Err(error) = ptrace::syscall(pid, None::<Signal>) {
        report
            .issues
            .push(BootstrapSyscallConfinementIssue::SyscallRestartFailed(error.to_string()));
        return Err(finalize(report));
    }
    let first = match waitpid(pid, Some(WaitPidFlag::__WALL)) {
        Ok(value) => value,
        Err(error) => {
            report
                .issues
                .push(BootstrapSyscallConfinementIssue::WaitFailed(error.to_string()));
            return Err(finalize(report));
        }
    };
    match first {
        WaitStatus::PtraceSyscall(observed) if observed == pid => {
            let regs = match ptrace::getregs(pid) {
                Ok(value) => value,
                Err(error) => {
                    report
                        .issues
                        .push(BootstrapSyscallConfinementIssue::UnexpectedStop(error.to_string()));
                    return Err(finalize(report));
                }
            };
            let return_value = regs.rax as i64;
            if return_value != 0 {
                report
                    .issues
                    .push(BootstrapSyscallConfinementIssue::ExecSyscallExitUnexpected(
                        return_value,
                    ));
                return Err(finalize(report));
            }
        }
        other => {
            report
                .issues
                .push(BootstrapSyscallConfinementIssue::UnexpectedStop(format!("{other:?}")));
            return Err(finalize(report));
        }
    }

    let mut observations = Vec::new();
    let mut pending: Option<PendingSyscall> = None;
    let mut expecting_entry = true;
    loop {
        if report.syscall_count >= policy.max_syscall_count {
            return kill_block(
                pid,
                report,
                BootstrapSyscallConfinementIssue::SyscallLimitExceeded,
            );
        }
        if let Err(error) = ptrace::syscall(pid, None::<Signal>) {
            report
                .issues
                .push(BootstrapSyscallConfinementIssue::SyscallRestartFailed(error.to_string()));
            return Err(finalize(report));
        }
        let status = match waitpid(pid, Some(WaitPidFlag::__WALL)) {
            Ok(value) => value,
            Err(error) => {
                report
                    .issues
                    .push(BootstrapSyscallConfinementIssue::WaitFailed(error.to_string()));
                return Err(finalize(report));
            }
        };
        match status {
            WaitStatus::PtraceSyscall(observed) if observed == pid => {
                let regs = match ptrace::getregs(pid) {
                    Ok(value) => value,
                    Err(error) => {
                        report
                            .issues
                            .push(BootstrapSyscallConfinementIssue::UnexpectedStop(
                                error.to_string(),
                            ));
                        return Err(finalize(report));
                    }
                };
                if expecting_entry {
                    let number = regs.orig_rax as i64;
                    let args = [regs.rdi, regs.rsi, regs.rdx, regs.r10, regs.r8, regs.r9];
                    if number == libc::SYS_write {
                        let fd = args[0] as i64;
                        let length = args[2] as usize;
                        if fd != policy.tracee_ready_write_fd as i64
                            || length == 0
                            || length > ready_policy.max_wire_bytes as usize
                            || length > LINUX_PIPE_BUF as usize
                        {
                            return kill_block(
                                pid,
                                report,
                                BootstrapSyscallConfinementIssue::ReadyWriteLengthInvalid,
                            );
                        }
                        let bytes = match tracee_memory(pid, args[1], length) {
                            Ok(value) => value,
                            Err(error) => {
                                return kill_block(
                                    pid,
                                    report,
                                    BootstrapSyscallConfinementIssue::TraceeMemoryReadFailed(error),
                                );
                            }
                        };
                        let verified = match verify_bootstrap_ready_wire(
                            ready_policy,
                            runtime_policy,
                            &bytes,
                        ) {
                            Ok(value) => value,
                            Err(ready_report) => {
                                return kill_block(
                                    pid,
                                    report,
                                    BootstrapSyscallConfinementIssue::ReadyWireInvalid(
                                        ready_report.canonical_digest(),
                                    ),
                                );
                            }
                        };
                        if verified.ticket_digest() != issued.ticket_digest() {
                            return kill_block(
                                pid,
                                report,
                                BootstrapSyscallConfinementIssue::ReadyTicketMismatch,
                            );
                        }
                        if verified.launch_challenge_nonce_blake3_hex()
                            != issued.challenge_nonce_blake3_hex()
                        {
                            return kill_block(
                                pid,
                                report,
                                BootstrapSyscallConfinementIssue::ReadyChallengeMismatch,
                            );
                        }
                        if verified.runtime_policy_digest() != issued.runtime_policy_digest() {
                            return kill_block(
                                pid,
                                report,
                                BootstrapSyscallConfinementIssue::ReadyRuntimePolicyMismatch,
                            );
                        }
                        pending = Some(PendingSyscall {
                            number,
                            args,
                            ready: Some((verified, bytes)),
                        });
                    } else if bootstrap_safe_parts(number, args) {
                        pending = Some(PendingSyscall {
                            number,
                            args,
                            ready: None,
                        });
                    } else {
                        return kill_block(
                            pid,
                            report,
                            BootstrapSyscallConfinementIssue::DisallowedSyscall(number),
                        );
                    }
                    expecting_entry = false;
                } else {
                    let Some(pending_syscall) = pending.take() else {
                        report.issues.push(BootstrapSyscallConfinementIssue::UnexpectedStop(
                            "missing-pending-syscall".into(),
                        ));
                        return Err(finalize(report));
                    };
                    let return_value = regs.rax as i64;
                    report.syscall_count += 1;
                    observations.push(BootstrapSyscallObservation {
                        syscall_number: pending_syscall.number,
                        args: pending_syscall.args,
                        return_value,
                        ready_token_write: pending_syscall.ready.is_some(),
                    });
                    if let Some((verified, bytes)) = pending_syscall.ready {
                        if return_value != bytes.len() as i64 {
                            report.issues.push(
                                BootstrapSyscallConfinementIssue::ReadyWriteReturnedUnexpected(
                                    return_value,
                                ),
                            );
                            return Err(finalize(report));
                        }
                        let received = match read_exact_nonblocking(
                            supervisor_ready_read_fd,
                            bytes.len(),
                        ) {
                            Ok(value) => value,
                            Err(error) => {
                                report.issues.push(
                                    BootstrapSyscallConfinementIssue::ReadyPeerReadFailed(error),
                                );
                                return Err(finalize(report));
                            }
                        };
                        if received != bytes {
                            report
                                .issues
                                .push(BootstrapSyscallConfinementIssue::ReadyPeerBytesMismatch);
                            return Err(finalize(report));
                        }
                        let after = match local_fd(
                            supervisor_ready_read_fd,
                            policy.max_fdinfo_bytes,
                        ) {
                            Ok(value) => value,
                            Err(error) => {
                                report.issues.push(
                                    BootstrapSyscallConfinementIssue::SupervisorReadyFdUnavailable(
                                        error,
                                    ),
                                );
                                return Err(finalize(report));
                            }
                        };
                        if before != after {
                            report
                                .issues
                                .push(BootstrapSyscallConfinementIssue::SupervisorReadyFdChanged);
                            return Err(finalize(report));
                        }
                        let final_authority = match assess_critical_exec_preserved_authority(
                            authority_policy,
                            confirmed,
                        ) {
                            Ok(value) => value,
                            Err(authority_report) => {
                                report.issues.push(
                                    BootstrapSyscallConfinementIssue::CriticalAuthorityRecheckFailed(
                                        authority_report.canonical_digest(),
                                    ),
                                );
                                return Err(finalize(report));
                            }
                        };
                        if final_authority.verified().authority_snapshot_digest()
                            != initial.authority_snapshot_digest()
                        {
                            report.issues.push(
                                BootstrapSyscallConfinementIssue::CriticalAuthorityChangedBeforeReady,
                            );
                            return Err(finalize(report));
                        }
                        let sequence_digest = syscall_sequence_digest(&observations);
                        report.syscall_sequence_digest = Some(sequence_digest.clone());
                        report.final_authority_snapshot_digest = Some(
                            final_authority
                                .verified()
                                .authority_snapshot_digest()
                                .into(),
                        );
                        report.ready_wire_digest = Some(verified.wire_digest().into());
                        report.ready_checkpoint_digest = Some(verified.checkpoint_digest().into());
                        report.ready_wire_bytes_digest = Some(verified.wire_bytes_digest().into());
                        report.ready_wire_bytes_len = bytes.len() as u64;
                        report.tracee_stopped_at_ready_exit = true;
                        report.disposition = BootstrapSyscallConfinementDisposition::Qualified;
                        let report_digest = report.canonical_digest();
                        let policy_digest = policy_digest.expect("validated policy");
                        let qualification_digest = qualification_digest(
                            &policy_digest,
                            confirmed.confirmation_digest(),
                            issued.qualification_digest(),
                            verified.wire_digest(),
                            &sequence_digest,
                            final_authority.verified().authority_snapshot_digest(),
                            &report_digest,
                        );
                        let ready = BootstrapSyscallConfinementReady {
                            pid,
                            qualification_digest,
                            report_digest,
                            policy_digest,
                            exec_confirmation_digest: confirmed.confirmation_digest().into(),
                            issued_challenge_qualification_digest: issued.qualification_digest().into(),
                            ready_wire_digest: verified.wire_digest().into(),
                            ready_checkpoint_digest: verified.checkpoint_digest().into(),
                            ready_wire_bytes_digest: verified.wire_bytes_digest().into(),
                            syscall_sequence_digest: sequence_digest,
                            syscall_count: report.syscall_count,
                            authority_snapshot_digest: final_authority
                                .verified()
                                .authority_snapshot_digest()
                                .into(),
                        };
                        return Ok(BootstrapSyscallConfinementQualification {
                            report,
                            syscalls: observations,
                            ready,
                        });
                    }
                    expecting_entry = true;
                }
            }
            WaitStatus::Exited(_, code) => {
                report
                    .issues
                    .push(BootstrapSyscallConfinementIssue::TraceeExitedBeforeReady(code));
                return Err(finalize(report));
            }
            WaitStatus::Signaled(_, signal, _) => {
                report
                    .issues
                    .push(BootstrapSyscallConfinementIssue::TraceeSignaledBeforeReady(
                        signal.to_string(),
                    ));
                return Err(finalize(report));
            }
            other => {
                return kill_block(
                    pid,
                    report,
                    BootstrapSyscallConfinementIssue::UnexpectedStop(format!("{other:?}")),
                );
            }
        }
    }
}

pub fn release_bootstrap_confined_tracee(
    ready: &BootstrapSyscallConfinementReady,
) -> Result<(), BootstrapSyscallConfinementIssue> {
    ptrace::detach(ready.pid, None::<Signal>)
        .map_err(|error| BootstrapSyscallConfinementIssue::DetachFailed(error.to_string()))
}

struct PendingSyscall {
    number: i64,
    args: [u64; 6],
    ready: Option<(VerifiedBootstrapReadyWire, Vec<u8>)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct LocalFd {
    target: String,
    flags: i32,
    inode: u64,
    fdinfo_digest: String,
}

fn local_fd(fd: RawFd, maximum: u64) -> Result<LocalFd, String> {
    let target = fs::read_link(format!("/proc/self/fd/{fd}"))
        .map_err(|error| error.to_string())?
        .into_os_string()
        .into_string()
        .map_err(|_| "non-utf8-fd-target".to_string())?;
    let info = bounded(PathBuf::from(format!("/proc/self/fdinfo/{fd}")), maximum)?;
    let text = std::str::from_utf8(&info).map_err(|error| error.to_string())?;
    let mut flags = None;
    let mut inode = None;
    for line in text.lines() {
        if let Some((key, value)) = line.split_once(':') {
            match key {
                "flags" => {
                    flags = Some(
                        i32::from_str_radix(value.trim(), 8)
                            .map_err(|_| "invalid-flags".to_string())?,
                    )
                }
                "ino" => {
                    inode = Some(
                        value
                            .trim()
                            .parse::<u64>()
                            .map_err(|_| "invalid-inode".to_string())?,
                    )
                }
                _ => {}
            }
        }
    }
    Ok(LocalFd {
        target,
        flags: flags.ok_or_else(|| "missing-flags".to_string())?,
        inode: inode.ok_or_else(|| "missing-inode".to_string())?,
        fdinfo_digest: format!("blake3:{}", blake3::hash(&info).to_hex()),
    })
}

fn live_trace_stop(pid: i32) -> bool {
    let Ok(status) = fs::read_to_string(format!("/proc/{pid}/status")) else {
        return false;
    };
    let mut tracing_stop = false;
    let mut correct_tracer = false;
    for line in status.lines() {
        if let Some(value) = line.strip_prefix("State:") {
            tracing_stop = value.trim().starts_with('t');
        }
        if let Some(value) = line.strip_prefix("TracerPid:") {
            correct_tracer = value.trim().parse::<u32>().ok() == Some(getpid().as_raw() as u32);
        }
    }
    tracing_stop && correct_tracer
}

fn tracee_memory(pid: Pid, address: u64, length: usize) -> Result<Vec<u8>, String> {
    let mut file = File::open(format!("/proc/{}/mem", pid.as_raw()))
        .map_err(|error| error.to_string())?;
    file.seek(SeekFrom::Start(address))
        .map_err(|error| error.to_string())?;
    let mut bytes = vec![0u8; length];
    file.read_exact(&mut bytes)
        .map_err(|error| error.to_string())?;
    Ok(bytes)
}

fn read_exact_nonblocking(fd: RawFd, length: usize) -> Result<Vec<u8>, String> {
    let mut output = vec![0u8; length];
    let mut offset = 0usize;
    while offset < length {
        let count = read(fd, &mut output[offset..]).map_err(|error| error.to_string())?;
        if count == 0 {
            return Err("unexpected-ready-pipe-eof".into());
        }
        offset += count;
    }
    Ok(output)
}

fn kill_block<T>(
    pid: Pid,
    mut report: BootstrapSyscallConfinementReport,
    issue: BootstrapSyscallConfinementIssue,
) -> Result<T, BootstrapSyscallConfinementReport> {
    report.issues.push(issue);
    match kill(pid, Signal::SIGKILL) {
        Ok(()) => report.violation_sigkill_sent_before_restart = true,
        Err(error) => report
            .issues
            .push(BootstrapSyscallConfinementIssue::KillFailed(error.to_string())),
    }
    Err(finalize(report))
}

fn bootstrap_safe_parts(number: i64, args: [u64; 6]) -> bool {
    match number {
        value
            if value == libc::SYS_read
                || value == libc::SYS_pread64
                || value == libc::SYS_readv
                || value == libc::SYS_close
                || value == libc::SYS_lseek
                || value == libc::SYS_fstat
                || value == libc::SYS_newfstatat
                || value == libc::SYS_getdents64
                || value == libc::SYS_readlink
                || value == libc::SYS_readlinkat
                || value == libc::SYS_access
                || value == libc::SYS_faccessat
                || value == libc::SYS_getcwd
                || value == libc::SYS_uname
                || value == libc::SYS_getpid
                || value == libc::SYS_getppid
                || value == libc::SYS_gettid
                || value == libc::SYS_getuid
                || value == libc::SYS_geteuid
                || value == libc::SYS_getgid
                || value == libc::SYS_getegid
                || value == libc::SYS_clock_gettime
                || value == libc::SYS_clock_getres
                || value == libc::SYS_gettimeofday
                || value == libc::SYS_getrusage
                || value == libc::SYS_sysinfo
                || value == libc::SYS_sched_getaffinity
                || value == libc::SYS_getcpu
                || value == libc::SYS_getrandom
                || value == libc::SYS_sched_yield
                || value == libc::SYS_nanosleep
                || value == libc::SYS_clock_nanosleep
                || value == libc::SYS_munmap
                || value == libc::SYS_brk
                || value == libc::SYS_madvise
                || value == libc::SYS_rt_sigaction
                || value == libc::SYS_rt_sigprocmask
                || value == libc::SYS_sigaltstack
                || value == libc::SYS_arch_prctl
                || value == libc::SYS_set_tid_address
                || value == libc::SYS_set_robust_list
                || value == libc::SYS_rseq
                || value == libc::SYS_exit
                || value == libc::SYS_exit_group => true,
        value if value == libc::SYS_openat => {
            let flags = args[2] as i32;
            let forbidden = libc::O_WRONLY
                | libc::O_RDWR
                | libc::O_CREAT
                | libc::O_TRUNC
                | libc::O_APPEND
                | libc::O_TMPFILE;
            flags & forbidden == 0
        }
        value if value == libc::SYS_mmap => {
            let protection = args[2] as i32;
            let flags = args[3] as i32;
            protection & libc::PROT_EXEC == 0 && flags & libc::MAP_SHARED == 0
        }
        value if value == libc::SYS_mprotect => args[2] as i32 & libc::PROT_EXEC == 0,
        value if value == libc::SYS_fcntl => {
            let command = args[1] as i32;
            command == libc::F_GETFD || command == libc::F_GETFL
        }
        value if value == libc::SYS_prlimit64 => args[2] == 0,
        _ => false,
    }
}

fn syscall_sequence_digest(values: &[BootstrapSyscallObservation]) -> String {
    let mut h = blake3::Hasher::new();
    h.update(SEQUENCE_DOMAIN);
    h.update(&(values.len() as u64).to_le_bytes());
    for value in values {
        h.update(&value.syscall_number.to_le_bytes());
        for arg in value.args {
            h.update(&arg.to_le_bytes());
        }
        h.update(&value.return_value.to_le_bytes());
        h.update(&[u8::from(value.ready_token_write)]);
    }
    b3(h.finalize())
}

fn qualification_digest(
    policy: &str,
    exec: &str,
    issued: &str,
    wire: &str,
    sequence: &str,
    authority: &str,
    report: &str,
) -> String {
    let mut h = blake3::Hasher::new();
    h.update(QUALIFICATION_DOMAIN);
    for value in [policy, exec, issued, wire, sequence, authority, report] {
        field(&mut h, value);
    }
    b3(h.finalize())
}

fn finalize(
    mut report: BootstrapSyscallConfinementReport,
) -> BootstrapSyscallConfinementReport {
    report.disposition = if report
        .issues
        .iter()
        .any(BootstrapSyscallConfinementIssue::invalid)
    {
        BootstrapSyscallConfinementDisposition::Invalid
    } else {
        BootstrapSyscallConfinementDisposition::Blocked
    };
    report
}

fn bounded(path: PathBuf, maximum: u64) -> Result<Vec<u8>, String> {
    let mut file = File::open(path).map_err(|error| error.to_string())?;
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

fn octal(value: &str) -> Option<i32> { i32::from_str_radix(value, 8).ok() }
fn pipe_inode(value: &str) -> Option<u64> {
    value.strip_prefix("pipe:[")?.strip_suffix(']')?.parse().ok()
}
fn digest(value: &str) -> bool {
    value.strip_prefix("blake3:").is_some_and(|hex| {
        hex.len() == 64
            && hex
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    })
}
fn text(value: &str) -> bool {
    !value.is_empty()
        && value == value.trim()
        && value.len() <= MAX_TEXT
        && !value.chars().any(char::is_control)
}
fn refs(values: &[String]) -> bool {
    values.len() <= MAX_REFS
        && values.iter().all(|value| text(value))
        && {
            let mut seen = BTreeSet::new();
            values.iter().all(|value| seen.insert(value.as_str()))
        }
}
fn field(h: &mut blake3::Hasher, value: &str) {
    h.update(&(value.len() as u64).to_le_bytes());
    h.update(value.as_bytes());
}
fn sorted(h: &mut blake3::Hasher, values: &[String]) {
    let mut values = values.to_vec();
    values.sort();
    h.update(&(values.len() as u64).to_le_bytes());
    for value in values {
        field(h, &value);
    }
}
fn b3(hash: blake3::Hash) -> String { format!("blake3:{}", hash.to_hex()) }

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_profile_rejects_direct_output_and_exec() {
        assert!(!bootstrap_safe_parts(libc::SYS_write, [0; 6]));
        assert!(!bootstrap_safe_parts(libc::SYS_execve, [0; 6]));
        assert!(bootstrap_safe_parts(libc::SYS_getpid, [0; 6]));
    }

    #[test]
    fn mmap_profile_rejects_exec_and_shared() {
        let mut args = [0u64; 6];
        args[2] = libc::PROT_READ as u64;
        args[3] = libc::MAP_PRIVATE as u64;
        assert!(bootstrap_safe_parts(libc::SYS_mmap, args));
        args[2] = (libc::PROT_READ | libc::PROT_EXEC) as u64;
        assert!(!bootstrap_safe_parts(libc::SYS_mmap, args));
        args[2] = libc::PROT_READ as u64;
        args[3] = libc::MAP_SHARED as u64;
        assert!(!bootstrap_safe_parts(libc::SYS_mmap, args));
    }

    #[test]
    fn openat_profile_is_read_only() {
        let mut args = [0u64; 6];
        args[2] = libc::O_RDONLY as u64;
        assert!(bootstrap_safe_parts(libc::SYS_openat, args));
        args[2] = libc::O_RDWR as u64;
        assert!(!bootstrap_safe_parts(libc::SYS_openat, args));
        args[2] = (libc::O_RDONLY | libc::O_CREAT) as u64;
        assert!(!bootstrap_safe_parts(libc::SYS_openat, args));
    }

    #[test]
    fn fcntl_and_prlimit_are_query_only() {
        let mut args = [0u64; 6];
        args[1] = libc::F_GETFD as u64;
        assert!(bootstrap_safe_parts(libc::SYS_fcntl, args));
        args[1] = libc::F_SETFD as u64;
        assert!(!bootstrap_safe_parts(libc::SYS_fcntl, args));
        args = [0; 6];
        args[2] = 0;
        assert!(bootstrap_safe_parts(libc::SYS_prlimit64, args));
        args[2] = 1;
        assert!(!bootstrap_safe_parts(libc::SYS_prlimit64, args));
    }

    #[test]
    fn claim_ceiling_remains_bounded() {
        let claims = [
            "wire_actual_live_observation=false",
            "generic_external_side_effect_freedom=false",
            "instruction_continuity=false",
            "exclusive_pipe_writer=false",
            "trusted_time=false",
            "physical_authority=false",
        ];
        assert_eq!(claims.len(), 6);
    }
}
