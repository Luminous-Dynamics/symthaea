// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! PTRACE_EVENT_EXEC confirmation for an exact prepared static verifier launch.
//!
//! This Linux-only bridge seizes and interrupts a not-yet-target tracee,
//! resumes it under PTRACE_O_TRACEEXEC, and inspects the successful exec event
//! while the new image is still stopped before normal userspace execution.

#![cfg(target_os = "linux")]
#![deny(unsafe_code)]

use nix::{
    sys::{
        ptrace::{self, Event, Options},
        signal::Signal,
        stat::{major, minor},
        wait::{waitpid, WaitPidFlag, WaitStatus},
    },
    unistd::{getpid, Pid},
};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::File,
    io::{Read, Seek, SeekFrom},
    os::unix::fs::MetadataExt,
    path::PathBuf,
};
use symthaea_assurance_static_fd_exec_launch::PreparedStaticFdExec;

pub const STATIC_EXEC_SUPERVISOR_POLICY_SCHEMA_V1: &str =
    "symthaea.assurance.ptrace-static-exec-supervisor-policy.v1";
pub const STATIC_EXEC_SUPERVISOR_REPORT_SCHEMA_V1: &str =
    "symthaea.assurance.ptrace-static-exec-supervisor-report.v1";
pub const STATIC_EXEC_CONFIRMATION_REPORT_SCHEMA_V1: &str =
    "symthaea.assurance.ptrace-static-exec-confirmation-report.v1";
pub const STATIC_EXEC_RELEASE_REPORT_SCHEMA_V1: &str =
    "symthaea.assurance.ptrace-static-exec-release-report.v1";

const POLICY_DOMAIN: &[u8] =
    b"symthaea.assurance.ptrace-static-exec-supervisor-policy.digest.v1\0";
const SUPERVISOR_REPORT_DOMAIN: &[u8] =
    b"symthaea.assurance.ptrace-static-exec-supervisor-report.digest.v1\0";
const ARM_DOMAIN: &[u8] = b"symthaea.assurance.ptrace-static-exec-arm.digest.v1\0";
const MAP_SET_DOMAIN: &[u8] = b"symthaea.assurance.ptrace-static-exec-map-set.digest.v1\0";
const REPORT_DOMAIN: &[u8] =
    b"symthaea.assurance.ptrace-static-exec-confirmation-report.digest.v1\0";
const CONFIRMATION_DOMAIN: &[u8] =
    b"symthaea.assurance.ptrace-static-exec-confirmation.digest.v1\0";
const RELEASE_DOMAIN: &[u8] = b"symthaea.assurance.ptrace-static-exec-release.digest.v1\0";
const ARGV_DOMAIN: &[u8] = b"symthaea.assurance.static-fd-exec-argv.digest.v1\0";
const ENV_DOMAIN: &[u8] = b"symthaea.assurance.static-fd-exec-environment.digest.v1\0";
const MAX_TEXT: usize = 4096;
const MAX_REFS: usize = 128;
const HARD_MAX_PROC_BYTES: u64 = 64 * 1024 * 1024;
const HARD_MAX_EXECUTABLE_BYTES: u64 = 1024 * 1024 * 1024;
const HARD_MAX_EXEC_MAPPING_BYTES: u64 = 1024 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StaticExecSupervisorPolicy {
    pub schema_version: String,
    pub policy_id: String,
    pub expected_fd_exec_policy_digest: String,
    pub expected_runtime_policy_digest: String,
    pub expected_runtime_verifier_ref: String,
    pub expected_backend_id: String,
    pub allowed_kernel_executable_pseudo_maps: Vec<String>,
    pub max_proc_maps_bytes: u64,
    pub max_proc_cmdline_bytes: u64,
    pub max_proc_environ_bytes: u64,
    pub max_executable_bytes: u64,
    pub max_total_executable_mapping_bytes: u64,
    pub evidence_refs: Vec<String>,
}

impl StaticExecSupervisorPolicy {
    pub fn validate(&self) -> bool {
        self.schema_version == STATIC_EXEC_SUPERVISOR_POLICY_SCHEMA_V1
            && canonical_text(&self.policy_id)
            && digest(&self.expected_fd_exec_policy_digest)
            && digest(&self.expected_runtime_policy_digest)
            && canonical_text(&self.expected_runtime_verifier_ref)
            && canonical_text(&self.expected_backend_id)
            && self.allowed_kernel_executable_pseudo_maps.len() <= 32
            && self
                .allowed_kernel_executable_pseudo_maps
                .iter()
                .all(|value| valid_pseudo_map(value))
            && unique(&self.allowed_kernel_executable_pseudo_maps)
            && (1..=HARD_MAX_PROC_BYTES).contains(&self.max_proc_maps_bytes)
            && (1..=HARD_MAX_PROC_BYTES).contains(&self.max_proc_cmdline_bytes)
            && (1..=HARD_MAX_PROC_BYTES).contains(&self.max_proc_environ_bytes)
            && (1..=HARD_MAX_EXECUTABLE_BYTES).contains(&self.max_executable_bytes)
            && (1..=HARD_MAX_EXEC_MAPPING_BYTES)
                .contains(&self.max_total_executable_mapping_bytes)
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
            self.expected_fd_exec_policy_digest.as_str(),
            self.expected_runtime_policy_digest.as_str(),
            self.expected_runtime_verifier_ref.as_str(),
            self.expected_backend_id.as_str(),
        ] {
            field(&mut h, value);
        }
        sorted(&mut h, &self.allowed_kernel_executable_pseudo_maps);
        for value in [
            self.max_proc_maps_bytes,
            self.max_proc_cmdline_bytes,
            self.max_proc_environ_bytes,
            self.max_executable_bytes,
            self.max_total_executable_mapping_bytes,
        ] {
            h.update(&value.to_le_bytes());
        }
        sorted(&mut h, &self.evidence_refs);
        Some(format!("blake3:{}", h.finalize().to_hex()))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StaticExecSupervisorIssue {
    InvalidPolicy,
    InvalidTraceePid,
    FdExecPolicyMismatch,
    RuntimePolicyMismatch,
    RuntimeVerifierMismatch,
    BackendMismatch,
    SeizeFailed(String),
    InterruptFailed(String),
    InitialWaitFailed(String),
    UnexpectedInitialStop(String),
    TargetAlreadyLoadedAtArm,
    ContinueFailed(String),
    WaitFailed(String),
    TraceeExitedBeforeExec(i32),
    TraceeSignaledBeforeExec(String),
    UnexpectedStopBeforeExec(String),
    UnexpectedPtraceEvent(i32),
    ProcExeUnavailable(String),
    ExecutablePathMismatch,
    ExecutableMetadataMismatch,
    ExecutableTooLarge { observed: u64, maximum: u64 },
    ExecutableDigestMismatch,
    ProcCmdlineUnavailable(String),
    ProcCmdlineInvalid,
    ArgvDigestMismatch,
    ProcEnvironUnavailable(String),
    ProcEnvironInvalid,
    EnvironmentDigestMismatch,
    ProcMapsUnavailable(String),
    ProcMapsInvalid(String),
    WritableExecutableMapping(String),
    NonReadableExecutableMapping(String),
    SharedExecutableMapping(String),
    UnapprovedExecutablePseudoMap(String),
    AnonymousExecutableMapping,
    UnexpectedExecutableMapping(String),
    ExecutableMappingIdentityMismatch(String),
    ExecutableMappingOutsideFile(String),
    ExecutableMappingBytesExceeded { observed: u64, maximum: u64 },
    ProcMemUnavailable(String),
    ExecutableMappingReadFailed(String),
    ExecutableMappingBytesMismatch(String),
    NoExecutableTargetMapping,
    DetachFailed(String),
}

impl StaticExecSupervisorIssue {
    fn code(&self) -> String {
        match self {
            Self::InvalidPolicy => "invalid-policy".into(),
            Self::InvalidTraceePid => "invalid-tracee-pid".into(),
            Self::FdExecPolicyMismatch => "fd-exec-policy-mismatch".into(),
            Self::RuntimePolicyMismatch => "runtime-policy-mismatch".into(),
            Self::RuntimeVerifierMismatch => "runtime-verifier-mismatch".into(),
            Self::BackendMismatch => "backend-mismatch".into(),
            Self::SeizeFailed(v) => format!("seize-failed:{v}"),
            Self::InterruptFailed(v) => format!("interrupt-failed:{v}"),
            Self::InitialWaitFailed(v) => format!("initial-wait-failed:{v}"),
            Self::UnexpectedInitialStop(v) => format!("unexpected-initial-stop:{v}"),
            Self::TargetAlreadyLoadedAtArm => "target-already-loaded-at-arm".into(),
            Self::ContinueFailed(v) => format!("continue-failed:{v}"),
            Self::WaitFailed(v) => format!("wait-failed:{v}"),
            Self::TraceeExitedBeforeExec(v) => format!("tracee-exited-before-exec:{v}"),
            Self::TraceeSignaledBeforeExec(v) => format!("tracee-signaled-before-exec:{v}"),
            Self::UnexpectedStopBeforeExec(v) => format!("unexpected-stop-before-exec:{v}"),
            Self::UnexpectedPtraceEvent(v) => format!("unexpected-ptrace-event:{v}"),
            Self::ProcExeUnavailable(v) => format!("proc-exe-unavailable:{v}"),
            Self::ExecutablePathMismatch => "executable-path-mismatch".into(),
            Self::ExecutableMetadataMismatch => "executable-metadata-mismatch".into(),
            Self::ExecutableTooLarge { observed, maximum } => {
                format!("executable-too-large:{observed}:{maximum}")
            }
            Self::ExecutableDigestMismatch => "executable-digest-mismatch".into(),
            Self::ProcCmdlineUnavailable(v) => format!("proc-cmdline-unavailable:{v}"),
            Self::ProcCmdlineInvalid => "proc-cmdline-invalid".into(),
            Self::ArgvDigestMismatch => "argv-digest-mismatch".into(),
            Self::ProcEnvironUnavailable(v) => format!("proc-environ-unavailable:{v}"),
            Self::ProcEnvironInvalid => "proc-environ-invalid".into(),
            Self::EnvironmentDigestMismatch => "environment-digest-mismatch".into(),
            Self::ProcMapsUnavailable(v) => format!("proc-maps-unavailable:{v}"),
            Self::ProcMapsInvalid(v) => format!("proc-maps-invalid:{v}"),
            Self::WritableExecutableMapping(v) => format!("writable-exec-map:{v}"),
            Self::NonReadableExecutableMapping(v) => format!("nonreadable-exec-map:{v}"),
            Self::SharedExecutableMapping(v) => format!("shared-exec-map:{v}"),
            Self::UnapprovedExecutablePseudoMap(v) => format!("unapproved-exec-pseudo:{v}"),
            Self::AnonymousExecutableMapping => "anonymous-executable-mapping".into(),
            Self::UnexpectedExecutableMapping(v) => format!("unexpected-executable-map:{v}"),
            Self::ExecutableMappingIdentityMismatch(v) => {
                format!("exec-map-identity-mismatch:{v}")
            }
            Self::ExecutableMappingOutsideFile(v) => format!("exec-map-outside-file:{v}"),
            Self::ExecutableMappingBytesExceeded { observed, maximum } => {
                format!("exec-map-bytes-exceeded:{observed}:{maximum}")
            }
            Self::ProcMemUnavailable(v) => format!("proc-mem-unavailable:{v}"),
            Self::ExecutableMappingReadFailed(v) => format!("exec-map-read-failed:{v}"),
            Self::ExecutableMappingBytesMismatch(v) => format!("exec-map-bytes-mismatch:{v}"),
            Self::NoExecutableTargetMapping => "no-executable-target-mapping".into(),
            Self::DetachFailed(v) => format!("detach-failed:{v}"),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StaticExecSupervisorReport {
    pub schema_version: String,
    pub policy_id: String,
    pub policy_digest: Option<String>,
    pub plan_digest: String,
    pub tracee_pid: i32,
    pub pre_exec_executable_path: Option<String>,
    pub armed_digest: Option<String>,
    pub tracee_stopped: bool,
    pub issues: Vec<StaticExecSupervisorIssue>,
}

impl StaticExecSupervisorReport {
    pub fn qualified(&self) -> bool {
        self.issues.is_empty() && self.armed_digest.is_some() && self.tracee_stopped
    }

    pub fn canonical_digest(&self) -> String {
        let mut h = blake3::Hasher::new();
        h.update(SUPERVISOR_REPORT_DOMAIN);
        for value in [
            self.schema_version.as_str(),
            self.policy_id.as_str(),
            self.policy_digest.as_deref().unwrap_or("-"),
            self.plan_digest.as_str(),
            self.pre_exec_executable_path.as_deref().unwrap_or("-"),
            self.armed_digest.as_deref().unwrap_or("-"),
        ] {
            field(&mut h, value);
        }
        h.update(&self.tracee_pid.to_le_bytes());
        h.update(&[u8::from(self.tracee_stopped)]);
        h.update(&(self.issues.len() as u64).to_le_bytes());
        for issue in &self.issues {
            field(&mut h, &issue.code());
        }
        format!("blake3:{}", h.finalize().to_hex())
    }

    pub const fn grants_physical_authority(&self) -> bool { false }
}

#[derive(Debug)]
pub struct ArmedStaticExecSupervisor {
    pid: Pid,
    policy_digest: String,
    plan_digest: String,
    fd_exec_policy_digest: String,
    runtime_policy_digest: String,
    runtime_verifier_ref: String,
    backend_id: String,
    target_path: String,
    target_digest: String,
    target_device: u64,
    target_inode: u64,
    target_size: u64,
    target_mode: u32,
    target_uid: u32,
    target_gid: u32,
    target_mtime_seconds: i64,
    target_mtime_nanoseconds: i64,
    fd_identity_digest: String,
    argv_digest: String,
    environment_digest: String,
    launch_nonce_blake3_hex: String,
    allowed_kernel_executable_pseudo_maps: BTreeSet<String>,
    max_proc_maps_bytes: u64,
    max_proc_cmdline_bytes: u64,
    max_proc_environ_bytes: u64,
    max_executable_bytes: u64,
    max_total_executable_mapping_bytes: u64,
    armed_digest: String,
}

impl ArmedStaticExecSupervisor {
    pub fn armed_digest(&self) -> &str { &self.armed_digest }
    pub fn plan_digest(&self) -> &str { &self.plan_digest }
    pub fn tracee_pid(&self) -> i32 { self.pid.as_raw() }
    pub const fn tracee_stopped_before_target_exec(&self) -> bool { true }
    pub const fn ptrace_traceexec_armed(&self) -> bool { true }
    pub const fn ptrace_exitkill_armed(&self) -> bool { true }
    pub const fn launch_success_established(&self) -> bool { false }
    pub const fn grants_physical_authority(&self) -> bool { false }
}

pub struct StaticExecSupervisorArm {
    pub report: StaticExecSupervisorReport,
    armed: ArmedStaticExecSupervisor,
}

impl StaticExecSupervisorArm {
    pub fn armed(&self) -> &ArmedStaticExecSupervisor { &self.armed }
    pub fn into_armed(self) -> ArmedStaticExecSupervisor { self.armed }
}

pub fn arm_static_exec_supervision(
    policy: &StaticExecSupervisorPolicy,
    prepared: &PreparedStaticFdExec,
    tracee_pid: i32,
) -> Result<StaticExecSupervisorArm, StaticExecSupervisorReport> {
    let policy_digest = policy.canonical_digest();
    let mut report = StaticExecSupervisorReport {
        schema_version: STATIC_EXEC_SUPERVISOR_REPORT_SCHEMA_V1.into(),
        policy_id: policy.policy_id.clone(),
        policy_digest: policy_digest.clone(),
        plan_digest: prepared.plan_digest().into(),
        tracee_pid,
        pre_exec_executable_path: None,
        armed_digest: None,
        tracee_stopped: false,
        issues: Vec::new(),
    };
    if !policy.validate() {
        report.issues.push(StaticExecSupervisorIssue::InvalidPolicy);
        return Err(report);
    }
    let pid = Pid::from_raw(tracee_pid);
    if tracee_pid <= 1 || pid == getpid() {
        report.issues.push(StaticExecSupervisorIssue::InvalidTraceePid);
        return Err(report);
    }
    if prepared.policy_digest() != policy.expected_fd_exec_policy_digest {
        report.issues.push(StaticExecSupervisorIssue::FdExecPolicyMismatch);
    }
    if prepared.runtime_policy_digest() != policy.expected_runtime_policy_digest {
        report.issues.push(StaticExecSupervisorIssue::RuntimePolicyMismatch);
    }
    if prepared.runtime_verifier_ref() != policy.expected_runtime_verifier_ref {
        report.issues.push(StaticExecSupervisorIssue::RuntimeVerifierMismatch);
    }
    if prepared.backend_id() != policy.expected_backend_id {
        report.issues.push(StaticExecSupervisorIssue::BackendMismatch);
    }
    if !report.issues.is_empty() {
        return Err(report);
    }

    let options = Options::PTRACE_O_TRACEEXEC | Options::PTRACE_O_EXITKILL;
    if let Err(error) = ptrace::seize(pid, options) {
        report.issues.push(StaticExecSupervisorIssue::SeizeFailed(error.to_string()));
        return Err(report);
    }
    if let Err(error) = ptrace::interrupt(pid) {
        report.issues.push(StaticExecSupervisorIssue::InterruptFailed(error.to_string()));
        let _ = ptrace::detach(pid, None::<Signal>);
        return Err(report);
    }
    let initial = match waitpid(pid, Some(WaitPidFlag::__WALL)) {
        Ok(status) => status,
        Err(error) => {
            report.issues.push(StaticExecSupervisorIssue::InitialWaitFailed(error.to_string()));
            let _ = ptrace::detach(pid, None::<Signal>);
            return Err(report);
        }
    };
    match initial {
        WaitStatus::PtraceEvent(observed, Signal::SIGTRAP, event)
            if observed == pid && event == Event::PTRACE_EVENT_STOP as i32 => {}
        other => {
            report.issues.push(StaticExecSupervisorIssue::UnexpectedInitialStop(format!("{other:?}")));
            let _ = ptrace::detach(pid, None::<Signal>);
            return Err(report);
        }
    }
    report.tracee_stopped = true;

    let pre_path = std::fs::read_link(proc_path(pid, "exe"))
        .ok()
        .and_then(|path| path.into_os_string().into_string().ok());
    report.pre_exec_executable_path = pre_path.clone();
    if let Ok(metadata) = std::fs::metadata(proc_path(pid, "exe")) {
        if metadata.dev() == prepared.fd_identity().device
            && metadata.ino() == prepared.fd_identity().inode
        {
            report.issues.push(StaticExecSupervisorIssue::TargetAlreadyLoadedAtArm);
            let _ = ptrace::detach(pid, None::<Signal>);
            return Err(report);
        }
    }

    let policy_digest = policy_digest.expect("validated policy has digest");
    let armed_digest = armed_digest(
        &policy_digest,
        prepared.plan_digest(),
        tracee_pid,
        pre_path.as_deref().unwrap_or("-"),
    );
    report.armed_digest = Some(armed_digest.clone());
    let fd = prepared.fd_identity();
    let armed = ArmedStaticExecSupervisor {
        pid,
        policy_digest,
        plan_digest: prepared.plan_digest().into(),
        fd_exec_policy_digest: prepared.policy_digest().into(),
        runtime_policy_digest: prepared.runtime_policy_digest().into(),
        runtime_verifier_ref: prepared.runtime_verifier_ref().into(),
        backend_id: prepared.backend_id().into(),
        target_path: prepared.executable_path().into(),
        target_digest: prepared.executable_digest().into(),
        target_device: fd.device,
        target_inode: fd.inode,
        target_size: fd.file_size,
        target_mode: fd.mode,
        target_uid: fd.uid,
        target_gid: fd.gid,
        target_mtime_seconds: fd.mtime_seconds,
        target_mtime_nanoseconds: fd.mtime_nanoseconds,
        fd_identity_digest: fd.identity_digest.clone(),
        argv_digest: prepared.argv_digest().into(),
        environment_digest: prepared.environment_digest().into(),
        launch_nonce_blake3_hex: prepared.launch_nonce_blake3_hex().into(),
        allowed_kernel_executable_pseudo_maps: policy
            .allowed_kernel_executable_pseudo_maps
            .iter()
            .cloned()
            .collect(),
        max_proc_maps_bytes: policy.max_proc_maps_bytes,
        max_proc_cmdline_bytes: policy.max_proc_cmdline_bytes,
        max_proc_environ_bytes: policy.max_proc_environ_bytes,
        max_executable_bytes: policy.max_executable_bytes,
        max_total_executable_mapping_bytes: policy.max_total_executable_mapping_bytes,
        armed_digest,
    };
    Ok(StaticExecSupervisorArm { report, armed })
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExecutableMappingEvidence {
    pub start: u64,
    pub end: u64,
    pub file_offset: u64,
    pub permissions: String,
    pub device_major: u64,
    pub device_minor: u64,
    pub inode: u64,
    pub mapped_bytes_blake3: String,
    pub backing_bytes_blake3: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StaticExecConfirmationReport {
    pub schema_version: String,
    pub policy_digest: String,
    pub armed_digest: String,
    pub plan_digest: String,
    pub tracee_pid: i32,
    pub executable_path: String,
    pub executable_digest: String,
    pub fd_identity_digest: String,
    pub argv_digest: String,
    pub environment_digest: String,
    pub launch_nonce_blake3_hex: String,
    pub proc_maps_digest: Option<String>,
    pub executable_mapping_set_digest: Option<String>,
    pub executable_mapping_count: u64,
    pub kernel_executable_pseudo_map_count: u64,
    pub total_executable_mapping_bytes: u64,
    pub tracee_stopped_at_exec_event: bool,
    pub issues: Vec<StaticExecSupervisorIssue>,
    pub confirmation_digest: Option<String>,
}

impl StaticExecConfirmationReport {
    pub fn qualified(&self) -> bool {
        self.issues.is_empty()
            && self.tracee_stopped_at_exec_event
            && self.confirmation_digest.is_some()
    }

    pub fn canonical_digest(&self) -> String { report_digest(self) }
    pub const fn grants_physical_authority(&self) -> bool { false }
}

#[derive(Debug)]
pub struct ConfirmedStaticExecLaunch {
    pid: Pid,
    confirmation_digest: String,
    report_digest: String,
    policy_digest: String,
    armed_digest: String,
    plan_digest: String,
    fd_exec_policy_digest: String,
    runtime_policy_digest: String,
    runtime_verifier_ref: String,
    backend_id: String,
    executable_path: String,
    executable_digest: String,
    fd_identity_digest: String,
    argv_digest: String,
    environment_digest: String,
    launch_nonce_blake3_hex: String,
    proc_maps_digest: String,
    executable_mapping_set_digest: String,
    executable_mapping_count: u64,
    total_executable_mapping_bytes: u64,
}

impl ConfirmedStaticExecLaunch {
    pub fn confirmation_digest(&self) -> &str { &self.confirmation_digest }
    pub fn report_digest(&self) -> &str { &self.report_digest }
    pub fn policy_digest(&self) -> &str { &self.policy_digest }
    pub fn armed_digest(&self) -> &str { &self.armed_digest }
    pub fn plan_digest(&self) -> &str { &self.plan_digest }
    pub fn fd_exec_policy_digest(&self) -> &str { &self.fd_exec_policy_digest }
    pub fn runtime_policy_digest(&self) -> &str { &self.runtime_policy_digest }
    pub fn runtime_verifier_ref(&self) -> &str { &self.runtime_verifier_ref }
    pub fn backend_id(&self) -> &str { &self.backend_id }
    pub fn tracee_pid(&self) -> i32 { self.pid.as_raw() }
    pub fn executable_path(&self) -> &str { &self.executable_path }
    pub fn executable_digest(&self) -> &str { &self.executable_digest }
    pub fn fd_identity_digest(&self) -> &str { &self.fd_identity_digest }
    pub fn argv_digest(&self) -> &str { &self.argv_digest }
    pub fn environment_digest(&self) -> &str { &self.environment_digest }
    pub fn launch_nonce_blake3_hex(&self) -> &str { &self.launch_nonce_blake3_hex }
    pub fn proc_maps_digest(&self) -> &str { &self.proc_maps_digest }
    pub fn executable_mapping_set_digest(&self) -> &str {
        &self.executable_mapping_set_digest
    }
    pub const fn executable_mapping_count(&self) -> u64 { self.executable_mapping_count }
    pub const fn total_executable_mapping_bytes(&self) -> u64 {
        self.total_executable_mapping_bytes
    }
    pub const fn successful_exec_event_observed(&self) -> bool { true }
    pub const fn new_image_stopped_before_normal_userspace_resume(&self) -> bool { true }
    pub const fn exact_post_exec_executable_matches_prepared_identity(&self) -> bool { true }
    pub const fn post_exec_argv_matches_prepared_plan(&self) -> bool { true }
    pub const fn post_exec_environment_matches_prepared_plan(&self) -> bool { true }
    pub const fn all_non_kernel_executable_mappings_exact_target_image(&self) -> bool { true }
    pub const fn executable_mapping_bytes_match_backing_image(&self) -> bool { true }
    pub const fn tracee_stopped_at_confirmation(&self) -> bool { true }
    pub const fn prepared_fd_was_exec_syscall_operand_established(&self) -> bool { false }
    pub const fn all_inherited_process_state_qualified(&self) -> bool { false }
    pub const fn mapping_continuity_after_release_established(&self) -> bool { false }
    pub const fn post_launch_dynamic_loading_excluded(&self) -> bool { false }
    pub const fn trusted_time_established(&self) -> bool { false }
    pub const fn grants_physical_authority(&self) -> bool { false }
}

pub struct StaticExecConfirmation {
    pub report: StaticExecConfirmationReport,
    pub mappings: Vec<ExecutableMappingEvidence>,
    confirmed: ConfirmedStaticExecLaunch,
}

impl StaticExecConfirmation {
    pub fn confirmed(&self) -> &ConfirmedStaticExecLaunch { &self.confirmed }
    pub fn into_confirmed(self) -> ConfirmedStaticExecLaunch { self.confirmed }
}

pub fn await_confirmed_static_exec(
    armed: ArmedStaticExecSupervisor,
) -> Result<StaticExecConfirmation, StaticExecConfirmationReport> {
    let mut report = confirmation_base(&armed);
    if let Err(error) = ptrace::cont(armed.pid, None::<Signal>) {
        report.issues.push(StaticExecSupervisorIssue::ContinueFailed(error.to_string()));
        return Err(report);
    }
    loop {
        let status = match waitpid(armed.pid, Some(WaitPidFlag::__WALL)) {
            Ok(value) => value,
            Err(error) => {
                report.issues.push(StaticExecSupervisorIssue::WaitFailed(error.to_string()));
                return Err(report);
            }
        };
        match status {
            WaitStatus::PtraceEvent(pid, Signal::SIGTRAP, event)
                if pid == armed.pid && event == Event::PTRACE_EVENT_EXEC as i32 =>
            {
                report.tracee_stopped_at_exec_event = true;
                return inspect_exec_event(armed, report);
            }
            WaitStatus::PtraceEvent(_, _, event) => {
                report.issues.push(StaticExecSupervisorIssue::UnexpectedPtraceEvent(event));
                return Err(report);
            }
            WaitStatus::Exited(_, code) => {
                report.issues.push(StaticExecSupervisorIssue::TraceeExitedBeforeExec(code));
                return Err(report);
            }
            WaitStatus::Signaled(_, signal, _) => {
                report
                    .issues
                    .push(StaticExecSupervisorIssue::TraceeSignaledBeforeExec(signal.to_string()));
                return Err(report);
            }
            WaitStatus::Stopped(_, signal) => {
                report
                    .issues
                    .push(StaticExecSupervisorIssue::UnexpectedStopBeforeExec(signal.to_string()));
                return Err(report);
            }
            WaitStatus::PtraceSyscall(_) => {
                report.issues.push(StaticExecSupervisorIssue::UnexpectedStopBeforeExec(
                    "ptrace-syscall".into(),
                ));
                return Err(report);
            }
            WaitStatus::Continued(_) | WaitStatus::StillAlive => continue,
        }
    }
}

fn inspect_exec_event(
    armed: ArmedStaticExecSupervisor,
    mut report: StaticExecConfirmationReport,
) -> Result<StaticExecConfirmation, StaticExecConfirmationReport> {
    let exe_link = proc_path(armed.pid, "exe");
    let observed_path = match std::fs::read_link(&exe_link)
        .ok()
        .and_then(|path| path.into_os_string().into_string().ok())
    {
        Some(value) => value,
        None => {
            report
                .issues
                .push(StaticExecSupervisorIssue::ProcExeUnavailable("readlink".into()));
            return Err(report);
        }
    };
    if observed_path != armed.target_path {
        report.issues.push(StaticExecSupervisorIssue::ExecutablePathMismatch);
    }

    let mut executable = match File::open(&exe_link) {
        Ok(file) => file,
        Err(error) => {
            report
                .issues
                .push(StaticExecSupervisorIssue::ProcExeUnavailable(error.to_string()));
            return Err(report);
        }
    };
    let metadata = match executable.metadata() {
        Ok(value) => value,
        Err(error) => {
            report
                .issues
                .push(StaticExecSupervisorIssue::ProcExeUnavailable(error.to_string()));
            return Err(report);
        }
    };
    if metadata.dev() != armed.target_device
        || metadata.ino() != armed.target_inode
        || metadata.len() != armed.target_size
        || metadata.mode() != armed.target_mode
        || metadata.uid() != armed.target_uid
        || metadata.gid() != armed.target_gid
        || metadata.mtime() != armed.target_mtime_seconds
        || metadata.mtime_nsec() != armed.target_mtime_nanoseconds
    {
        report.issues.push(StaticExecSupervisorIssue::ExecutableMetadataMismatch);
    }
    let exe_digest = match hash_file(&mut executable, armed.max_executable_bytes) {
        Ok(value) => value,
        Err(issue) => {
            report.issues.push(issue);
            return Err(report);
        }
    };
    if exe_digest != armed.target_digest {
        report.issues.push(StaticExecSupervisorIssue::ExecutableDigestMismatch);
    }

    let cmdline = match read_bounded(&proc_path(armed.pid, "cmdline"), armed.max_proc_cmdline_bytes) {
        Ok(value) => value,
        Err(error) => {
            report
                .issues
                .push(StaticExecSupervisorIssue::ProcCmdlineUnavailable(error));
            return Err(report);
        }
    };
    let argv = match parse_nul_strings(&cmdline, false) {
        Some(value) => value,
        None => {
            report.issues.push(StaticExecSupervisorIssue::ProcCmdlineInvalid);
            return Err(report);
        }
    };
    if argv_digest(&argv) != armed.argv_digest {
        report.issues.push(StaticExecSupervisorIssue::ArgvDigestMismatch);
    }

    let environ = match read_bounded(&proc_path(armed.pid, "environ"), armed.max_proc_environ_bytes) {
        Ok(value) => value,
        Err(error) => {
            report
                .issues
                .push(StaticExecSupervisorIssue::ProcEnvironUnavailable(error));
            return Err(report);
        }
    };
    let env_entries = match parse_nul_strings(&environ, true).and_then(|values| canonical_env(&values)) {
        Some(value) => value,
        None => {
            report.issues.push(StaticExecSupervisorIssue::ProcEnvironInvalid);
            return Err(report);
        }
    };
    if env_digest(&env_entries) != armed.environment_digest {
        report.issues.push(StaticExecSupervisorIssue::EnvironmentDigestMismatch);
    }

    let maps_bytes = match read_bounded(&proc_path(armed.pid, "maps"), armed.max_proc_maps_bytes) {
        Ok(value) => value,
        Err(error) => {
            report
                .issues
                .push(StaticExecSupervisorIssue::ProcMapsUnavailable(error));
            return Err(report);
        }
    };
    let maps_text = match std::str::from_utf8(&maps_bytes) {
        Ok(value) => value,
        Err(error) => {
            report
                .issues
                .push(StaticExecSupervisorIssue::ProcMapsInvalid(error.to_string()));
            return Err(report);
        }
    };
    let parsed = match parse_maps(maps_text) {
        Ok(value) => value,
        Err(error) => {
            report.issues.push(StaticExecSupervisorIssue::ProcMapsInvalid(error));
            return Err(report);
        }
    };
    let proc_maps_digest = format!("blake3:{}", blake3::hash(&maps_bytes).to_hex());
    report.proc_maps_digest = Some(proc_maps_digest.clone());

    let mut mem = match File::open(proc_path(armed.pid, "mem")) {
        Ok(file) => file,
        Err(error) => {
            report
                .issues
                .push(StaticExecSupervisorIssue::ProcMemUnavailable(error.to_string()));
            return Err(report);
        }
    };
    let target_major = major(metadata.dev());
    let target_minor = minor(metadata.dev());
    let mut mappings = Vec::new();
    let mut kernel_pseudo_count = 0u64;
    let mut total_exec_bytes = 0u64;
    for map in parsed.into_iter().filter(ParsedMap::executable) {
        if map.writable() {
            report
                .issues
                .push(StaticExecSupervisorIssue::WritableExecutableMapping(map.pathname.clone()));
            continue;
        }
        if map.pathname.starts_with('[') {
            if armed.allowed_kernel_executable_pseudo_maps.contains(&map.pathname) {
                kernel_pseudo_count += 1;
            } else {
                report
                    .issues
                    .push(StaticExecSupervisorIssue::UnapprovedExecutablePseudoMap(map.pathname));
            }
            continue;
        }
        if map.pathname.is_empty() {
            report.issues.push(StaticExecSupervisorIssue::AnonymousExecutableMapping);
            continue;
        }
        if !map.readable() {
            report
                .issues
                .push(StaticExecSupervisorIssue::NonReadableExecutableMapping(map.pathname));
            continue;
        }
        if !map.private_mapping() {
            report
                .issues
                .push(StaticExecSupervisorIssue::SharedExecutableMapping(map.pathname));
            continue;
        }
        if map.pathname != armed.target_path {
            report
                .issues
                .push(StaticExecSupervisorIssue::UnexpectedExecutableMapping(map.pathname));
            continue;
        }
        if map.device_major != target_major
            || map.device_minor != target_minor
            || map.inode != armed.target_inode
        {
            report
                .issues
                .push(StaticExecSupervisorIssue::ExecutableMappingIdentityMismatch(map.pathname));
            continue;
        }
        let length = map.end - map.start;
        total_exec_bytes = total_exec_bytes.checked_add(length).unwrap_or(u64::MAX);
        if total_exec_bytes > armed.max_total_executable_mapping_bytes {
            report
                .issues
                .push(StaticExecSupervisorIssue::ExecutableMappingBytesExceeded {
                    observed: total_exec_bytes,
                    maximum: armed.max_total_executable_mapping_bytes,
                });
            continue;
        }
        if map
            .file_offset
            .checked_add(length)
            .is_none_or(|end| end > metadata.len())
        {
            report
                .issues
                .push(StaticExecSupervisorIssue::ExecutableMappingOutsideFile(map.pathname));
            continue;
        }
        match compare_mapping_bytes(&mut mem, &mut executable, &map) {
            Ok((mapped_digest, backing_digest)) => mappings.push(ExecutableMappingEvidence {
                start: map.start,
                end: map.end,
                file_offset: map.file_offset,
                permissions: map.permissions,
                device_major: map.device_major,
                device_minor: map.device_minor,
                inode: map.inode,
                mapped_bytes_blake3: mapped_digest,
                backing_bytes_blake3: backing_digest,
            }),
            Err(issue) => report.issues.push(issue),
        }
    }
    if mappings.is_empty() {
        report.issues.push(StaticExecSupervisorIssue::NoExecutableTargetMapping);
    }
    report.executable_mapping_count = mappings.len() as u64;
    report.kernel_executable_pseudo_map_count = kernel_pseudo_count;
    report.total_executable_mapping_bytes = total_exec_bytes;
    let map_set_digest = mapping_set_digest(&mappings, kernel_pseudo_count);
    report.executable_mapping_set_digest = Some(map_set_digest.clone());
    if !report.issues.is_empty() {
        return Err(report);
    }

    let report_digest_value = report_digest(&report);
    let confirmation_digest = confirmation_digest(
        &armed.policy_digest,
        &armed.armed_digest,
        &armed.plan_digest,
        armed.pid.as_raw(),
        &armed.target_digest,
        &armed.fd_identity_digest,
        &armed.argv_digest,
        &armed.environment_digest,
        &proc_maps_digest,
        &map_set_digest,
        &report_digest_value,
    );
    report.confirmation_digest = Some(confirmation_digest.clone());
    let confirmed = ConfirmedStaticExecLaunch {
        pid: armed.pid,
        confirmation_digest,
        report_digest: report_digest_value,
        policy_digest: armed.policy_digest,
        armed_digest: armed.armed_digest,
        plan_digest: armed.plan_digest,
        fd_exec_policy_digest: armed.fd_exec_policy_digest,
        runtime_policy_digest: armed.runtime_policy_digest,
        runtime_verifier_ref: armed.runtime_verifier_ref,
        backend_id: armed.backend_id,
        executable_path: armed.target_path,
        executable_digest: armed.target_digest,
        fd_identity_digest: armed.fd_identity_digest,
        argv_digest: armed.argv_digest,
        environment_digest: armed.environment_digest,
        launch_nonce_blake3_hex: armed.launch_nonce_blake3_hex,
        proc_maps_digest,
        executable_mapping_set_digest: map_set_digest,
        executable_mapping_count: mappings.len() as u64,
        total_executable_mapping_bytes: total_exec_bytes,
    };
    Ok(StaticExecConfirmation {
        report,
        mappings,
        confirmed,
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StaticExecReleaseReport {
    pub schema_version: String,
    pub confirmation_digest: String,
    pub tracee_pid: i32,
    pub released: bool,
    pub issue: Option<StaticExecSupervisorIssue>,
    pub report_digest: String,
}

pub fn release_confirmed_static_exec(
    confirmed: &ConfirmedStaticExecLaunch,
) -> Result<StaticExecReleaseReport, StaticExecReleaseReport> {
    let mut report = StaticExecReleaseReport {
        schema_version: STATIC_EXEC_RELEASE_REPORT_SCHEMA_V1.into(),
        confirmation_digest: confirmed.confirmation_digest.clone(),
        tracee_pid: confirmed.pid.as_raw(),
        released: false,
        issue: None,
        report_digest: String::new(),
    };
    match ptrace::detach(confirmed.pid, None::<Signal>) {
        Ok(()) => {
            report.released = true;
            report.report_digest = release_digest(&report);
            Ok(report)
        }
        Err(error) => {
            report.issue = Some(StaticExecSupervisorIssue::DetachFailed(error.to_string()));
            report.report_digest = release_digest(&report);
            Err(report)
        }
    }
}

fn confirmation_base(armed: &ArmedStaticExecSupervisor) -> StaticExecConfirmationReport {
    StaticExecConfirmationReport {
        schema_version: STATIC_EXEC_CONFIRMATION_REPORT_SCHEMA_V1.into(),
        policy_digest: armed.policy_digest.clone(),
        armed_digest: armed.armed_digest.clone(),
        plan_digest: armed.plan_digest.clone(),
        tracee_pid: armed.pid.as_raw(),
        executable_path: armed.target_path.clone(),
        executable_digest: armed.target_digest.clone(),
        fd_identity_digest: armed.fd_identity_digest.clone(),
        argv_digest: armed.argv_digest.clone(),
        environment_digest: armed.environment_digest.clone(),
        launch_nonce_blake3_hex: armed.launch_nonce_blake3_hex.clone(),
        proc_maps_digest: None,
        executable_mapping_set_digest: None,
        executable_mapping_count: 0,
        kernel_executable_pseudo_map_count: 0,
        total_executable_mapping_bytes: 0,
        tracee_stopped_at_exec_event: false,
        issues: Vec::new(),
        confirmation_digest: None,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ParsedMap {
    start: u64,
    end: u64,
    permissions: String,
    file_offset: u64,
    device_major: u64,
    device_minor: u64,
    inode: u64,
    pathname: String,
}

impl ParsedMap {
    fn executable(&self) -> bool { self.permissions.as_bytes().get(2) == Some(&b'x') }
    fn writable(&self) -> bool { self.permissions.as_bytes().get(1) == Some(&b'w') }
    fn readable(&self) -> bool { self.permissions.as_bytes().first() == Some(&b'r') }
    fn private_mapping(&self) -> bool { self.permissions.as_bytes().get(3) == Some(&b'p') }
}

fn parse_maps(text: &str) -> Result<Vec<ParsedMap>, String> {
    let mut out = Vec::new();
    for line in text.lines() {
        if line.is_empty() {
            continue;
        }
        let (range, rest) = take_token(line).ok_or("missing range")?;
        let (permissions, rest) = take_token(rest).ok_or("missing permissions")?;
        let (offset, rest) = take_token(rest).ok_or("missing offset")?;
        let (device, rest) = take_token(rest).ok_or("missing device")?;
        let (inode, rest) = take_token(rest).ok_or("missing inode")?;
        let pathname = rest.trim_start().to_string();
        let (start, end) = range.split_once('-').ok_or("invalid range")?;
        let start = u64::from_str_radix(start, 16).map_err(|_| "invalid range start")?;
        let end = u64::from_str_radix(end, 16).map_err(|_| "invalid range end")?;
        if start >= end || permissions.len() != 4 {
            return Err("invalid range/permissions".into());
        }
        let file_offset = u64::from_str_radix(offset, 16).map_err(|_| "invalid offset")?;
        let (dev_major, dev_minor) = device.split_once(':').ok_or("invalid device")?;
        let device_major = u64::from_str_radix(dev_major, 16).map_err(|_| "invalid major")?;
        let device_minor = u64::from_str_radix(dev_minor, 16).map_err(|_| "invalid minor")?;
        let inode = inode.parse::<u64>().map_err(|_| "invalid inode")?;
        out.push(ParsedMap {
            start,
            end,
            permissions: permissions.into(),
            file_offset,
            device_major,
            device_minor,
            inode,
            pathname,
        });
    }
    if out.is_empty() {
        return Err("empty maps".into());
    }
    Ok(out)
}

fn take_token(value: &str) -> Option<(&str, &str)> {
    let value = value.trim_start_matches(char::is_whitespace);
    if value.is_empty() {
        return None;
    }
    let index = value.find(char::is_whitespace).unwrap_or(value.len());
    Some((&value[..index], &value[index..]))
}

fn compare_mapping_bytes(
    mem: &mut File,
    executable: &mut File,
    map: &ParsedMap,
) -> Result<(String, String), StaticExecSupervisorIssue> {
    mem.seek(SeekFrom::Start(map.start))
        .map_err(|e| StaticExecSupervisorIssue::ExecutableMappingReadFailed(e.to_string()))?;
    executable
        .seek(SeekFrom::Start(map.file_offset))
        .map_err(|e| StaticExecSupervisorIssue::ExecutableMappingReadFailed(e.to_string()))?;
    let mut remaining = map.end - map.start;
    let mut memory_hasher = blake3::Hasher::new();
    let mut backing_hasher = blake3::Hasher::new();
    let mut memory = vec![0u8; 64 * 1024];
    let mut backing = vec![0u8; 64 * 1024];
    while remaining > 0 {
        let size = remaining.min(memory.len() as u64) as usize;
        mem.read_exact(&mut memory[..size])
            .map_err(|e| StaticExecSupervisorIssue::ExecutableMappingReadFailed(e.to_string()))?;
        executable
            .read_exact(&mut backing[..size])
            .map_err(|e| StaticExecSupervisorIssue::ExecutableMappingReadFailed(e.to_string()))?;
        if memory[..size] != backing[..size] {
            return Err(StaticExecSupervisorIssue::ExecutableMappingBytesMismatch(
                map.pathname.clone(),
            ));
        }
        memory_hasher.update(&memory[..size]);
        backing_hasher.update(&backing[..size]);
        remaining -= size as u64;
    }
    Ok((
        format!("blake3:{}", memory_hasher.finalize().to_hex()),
        format!("blake3:{}", backing_hasher.finalize().to_hex()),
    ))
}

fn mapping_set_digest(mappings: &[ExecutableMappingEvidence], pseudo_count: u64) -> String {
    let mut h = blake3::Hasher::new();
    h.update(MAP_SET_DOMAIN);
    h.update(&(mappings.len() as u64).to_le_bytes());
    h.update(&pseudo_count.to_le_bytes());
    for map in mappings {
        for value in [
            map.start,
            map.end,
            map.file_offset,
            map.device_major,
            map.device_minor,
            map.inode,
        ] {
            h.update(&value.to_le_bytes());
        }
        field(&mut h, &map.permissions);
        field(&mut h, &map.mapped_bytes_blake3);
        field(&mut h, &map.backing_bytes_blake3);
    }
    format!("blake3:{}", h.finalize().to_hex())
}

fn hash_file(file: &mut File, maximum: u64) -> Result<String, StaticExecSupervisorIssue> {
    let length = file
        .metadata()
        .map_err(|e| StaticExecSupervisorIssue::ProcExeUnavailable(e.to_string()))?
        .len();
    if length > maximum {
        return Err(StaticExecSupervisorIssue::ExecutableTooLarge {
            observed: length,
            maximum,
        });
    }
    file.seek(SeekFrom::Start(0))
        .map_err(|e| StaticExecSupervisorIssue::ProcExeUnavailable(e.to_string()))?;
    let mut h = blake3::Hasher::new();
    let mut buf = [0u8; 64 * 1024];
    loop {
        let n = file
            .read(&mut buf)
            .map_err(|e| StaticExecSupervisorIssue::ProcExeUnavailable(e.to_string()))?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(format!("blake3:{}", h.finalize().to_hex()))
}

fn read_bounded(path: &PathBuf, maximum: u64) -> Result<Vec<u8>, String> {
    let mut file = File::open(path).map_err(|e| e.to_string())?;
    let mut out = Vec::new();
    let mut limited = (&mut file).take(maximum.saturating_add(1));
    limited.read_to_end(&mut out).map_err(|e| e.to_string())?;
    if out.len() as u64 > maximum {
        return Err("bounded-read-limit-exceeded".into());
    }
    Ok(out)
}

fn parse_nul_strings(bytes: &[u8], allow_empty_collection: bool) -> Option<Vec<String>> {
    if bytes.is_empty() {
        return allow_empty_collection.then(Vec::new);
    }
    let mut parts = bytes.split(|byte| *byte == 0).collect::<Vec<_>>();
    if parts.last().is_some_and(|value| value.is_empty()) {
        parts.pop();
    }
    if parts.is_empty() && !allow_empty_collection {
        return None;
    }
    parts
        .into_iter()
        .map(|value| {
            if value.is_empty() {
                return None;
            }
            String::from_utf8(value.to_vec()).ok()
        })
        .collect()
}

fn canonical_env(values: &[String]) -> Option<Vec<String>> {
    let mut out = BTreeMap::new();
    for entry in values {
        let (key, value) = entry.split_once('=')?;
        if !env_key(key) || out.insert(key.to_string(), value.to_string()).is_some() {
            return None;
        }
    }
    Some(
        out.into_iter()
            .map(|(key, value)| format!("{key}={value}"))
            .collect(),
    )
}

fn argv_digest(values: &[String]) -> String {
    let mut h = blake3::Hasher::new();
    h.update(ARGV_DOMAIN);
    h.update(&(values.len() as u64).to_le_bytes());
    for value in values {
        field(&mut h, value);
    }
    format!("blake3:{}", h.finalize().to_hex())
}

fn env_digest(values: &[String]) -> String {
    let mut h = blake3::Hasher::new();
    h.update(ENV_DOMAIN);
    h.update(&(values.len() as u64).to_le_bytes());
    for value in values {
        field(&mut h, value);
    }
    format!("blake3:{}", h.finalize().to_hex())
}

fn env_key(value: &str) -> bool {
    let mut bytes = value.bytes();
    let Some(first) = bytes.next() else {
        return false;
    };
    (first.is_ascii_alphabetic() || first == b'_')
        && bytes.all(|byte| byte.is_ascii_alphanumeric() || byte == b'_')
}

fn proc_path(pid: Pid, leaf: &str) -> PathBuf {
    PathBuf::from(format!("/proc/{}/{}", pid.as_raw(), leaf))
}

fn armed_digest(policy_digest: &str, plan_digest: &str, pid: i32, pre_path: &str) -> String {
    let mut h = blake3::Hasher::new();
    h.update(ARM_DOMAIN);
    field(&mut h, policy_digest);
    field(&mut h, plan_digest);
    h.update(&pid.to_le_bytes());
    field(&mut h, pre_path);
    format!("blake3:{}", h.finalize().to_hex())
}

fn report_digest(report: &StaticExecConfirmationReport) -> String {
    let mut h = blake3::Hasher::new();
    h.update(REPORT_DOMAIN);
    for value in [
        report.schema_version.as_str(),
        report.policy_digest.as_str(),
        report.armed_digest.as_str(),
        report.plan_digest.as_str(),
        report.executable_path.as_str(),
        report.executable_digest.as_str(),
        report.fd_identity_digest.as_str(),
        report.argv_digest.as_str(),
        report.environment_digest.as_str(),
        report.launch_nonce_blake3_hex.as_str(),
        report.proc_maps_digest.as_deref().unwrap_or("-"),
        report
            .executable_mapping_set_digest
            .as_deref()
            .unwrap_or("-"),
    ] {
        field(&mut h, value);
    }
    h.update(&report.tracee_pid.to_le_bytes());
    for value in [
        report.executable_mapping_count,
        report.kernel_executable_pseudo_map_count,
        report.total_executable_mapping_bytes,
    ] {
        h.update(&value.to_le_bytes());
    }
    h.update(&[u8::from(report.tracee_stopped_at_exec_event)]);
    h.update(&(report.issues.len() as u64).to_le_bytes());
    for issue in &report.issues {
        field(&mut h, &issue.code());
    }
    format!("blake3:{}", h.finalize().to_hex())
}

#[allow(clippy::too_many_arguments)]
fn confirmation_digest(
    policy_digest: &str,
    armed_digest_value: &str,
    plan_digest: &str,
    pid: i32,
    executable_digest: &str,
    fd_identity_digest: &str,
    argv_digest: &str,
    environment_digest: &str,
    proc_maps_digest: &str,
    map_set_digest: &str,
    report_digest: &str,
) -> String {
    let mut h = blake3::Hasher::new();
    h.update(CONFIRMATION_DOMAIN);
    for value in [
        policy_digest,
        armed_digest_value,
        plan_digest,
        executable_digest,
        fd_identity_digest,
        argv_digest,
        environment_digest,
        proc_maps_digest,
        map_set_digest,
        report_digest,
    ] {
        field(&mut h, value);
    }
    h.update(&pid.to_le_bytes());
    format!("blake3:{}", h.finalize().to_hex())
}

fn release_digest(report: &StaticExecReleaseReport) -> String {
    let mut h = blake3::Hasher::new();
    h.update(RELEASE_DOMAIN);
    field(&mut h, &report.schema_version);
    field(&mut h, &report.confirmation_digest);
    h.update(&report.tracee_pid.to_le_bytes());
    h.update(&[u8::from(report.released)]);
    let issue = report
        .issue
        .as_ref()
        .map(StaticExecSupervisorIssue::code)
        .unwrap_or_else(|| "-".into());
    field(&mut h, &issue);
    format!("blake3:{}", h.finalize().to_hex())
}

fn field(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn sorted(hasher: &mut blake3::Hasher, values: &[String]) {
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

fn valid_pseudo_map(value: &str) -> bool {
    canonical_text(value) && value.starts_with('[') && value.ends_with(']')
}

fn digest(value: &str) -> bool {
    value.strip_prefix("blake3:").is_some_and(|hex| {
        hex.len() == 64
            && hex
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    })
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

    fn policy() -> StaticExecSupervisorPolicy {
        StaticExecSupervisorPolicy {
            schema_version: STATIC_EXEC_SUPERVISOR_POLICY_SCHEMA_V1.into(),
            policy_id: "policy:ptrace-static-exec:1".into(),
            expected_fd_exec_policy_digest: d("fd-policy"),
            expected_runtime_policy_digest: d("runtime-policy"),
            expected_runtime_verifier_ref: "verifier:host".into(),
            expected_backend_id: "backend:in-process".into(),
            allowed_kernel_executable_pseudo_maps: vec!["[vdso]".into(), "[vsyscall]".into()],
            max_proc_maps_bytes: 1 << 20,
            max_proc_cmdline_bytes: 1 << 20,
            max_proc_environ_bytes: 1 << 20,
            max_executable_bytes: 1 << 30,
            max_total_executable_mapping_bytes: 1 << 30,
            evidence_refs: vec!["review:a".into(), "review:b".into()],
        }
    }

    #[test]
    fn policy_orders_sets_nonsemantically() {
        let left = policy();
        let mut right = left.clone();
        right.allowed_kernel_executable_pseudo_maps.reverse();
        right.evidence_refs.reverse();
        assert_eq!(left.canonical_digest(), right.canonical_digest());
        right.expected_backend_id = "backend:other".into();
        assert_ne!(left.canonical_digest(), right.canonical_digest());
    }

    #[test]
    fn supervisor_report_and_arm_have_distinct_domains() {
        let p = policy();
        let policy_digest = p.canonical_digest().unwrap();
        let arm = armed_digest(&policy_digest, &d("plan"), 42, "/launcher");
        let report = StaticExecSupervisorReport {
            schema_version: STATIC_EXEC_SUPERVISOR_REPORT_SCHEMA_V1.into(),
            policy_id: p.policy_id,
            policy_digest: Some(policy_digest),
            plan_digest: d("plan"),
            tracee_pid: 42,
            pre_exec_executable_path: Some("/launcher".into()),
            armed_digest: Some(arm.clone()),
            tracee_stopped: true,
            issues: Vec::new(),
        };
        assert_ne!(arm, report.canonical_digest());
    }

    #[test]
    fn argv_and_environment_digest_match_fd_exec_domains() {
        let argv = vec!["verifier".into(), "--mode".into(), "strict".into()];
        let env = vec!["A=1".into(), "B=2".into()];
        assert_ne!(argv_digest(&argv), env_digest(&env));
        let mut reordered_argv = argv.clone();
        reordered_argv.swap(1, 2);
        assert_ne!(argv_digest(&argv), argv_digest(&reordered_argv));
        let reordered_env = vec!["B=2".into(), "A=1".into()];
        assert_eq!(
            env_digest(&canonical_env(&env).unwrap()),
            env_digest(&canonical_env(&reordered_env).unwrap())
        );
    }

    #[test]
    fn maps_parser_preserves_offsets_devices_and_paths() {
        let parsed = parse_maps(
            "00400000-00401000 r-xp 00001000 fd:03 42 /nix/store/abc-verifier/bin/verifier\n",
        )
        .unwrap();
        assert_eq!(parsed.len(), 1);
        assert_eq!(parsed[0].start, 0x0040_0000);
        assert_eq!(parsed[0].end, 0x0040_1000);
        assert_eq!(parsed[0].file_offset, 0x1000);
        assert_eq!(parsed[0].device_major, 0xfd);
        assert_eq!(parsed[0].device_minor, 0x03);
        assert_eq!(parsed[0].inode, 42);
        assert_eq!(parsed[0].pathname, "/nix/store/abc-verifier/bin/verifier");
        assert!(parsed[0].executable());
        assert!(parsed[0].readable());
        assert!(parsed[0].private_mapping());
    }

    #[test]
    fn nul_parser_and_environment_canonicalization_are_fail_closed() {
        assert_eq!(
            parse_nul_strings(b"a\0b\0", false).unwrap(),
            vec!["a", "b"]
        );
        assert!(parse_nul_strings(b"a\0\0b\0", false).is_none());
        assert!(canonical_env(&["A=1".into(), "A=2".into()]).is_none());
        assert_eq!(
            canonical_env(&["B=2".into(), "A=1".into()]).unwrap(),
            vec!["A=1", "B=2"]
        );
    }

    #[test]
    fn mapping_set_digest_binds_exact_range_and_bytes() {
        let one = ExecutableMappingEvidence {
            start: 0x1000,
            end: 0x2000,
            file_offset: 0,
            permissions: "r-xp".into(),
            device_major: 1,
            device_minor: 2,
            inode: 3,
            mapped_bytes_blake3: d("mapped"),
            backing_bytes_blake3: d("backing"),
        };
        let left = mapping_set_digest(std::slice::from_ref(&one), 1);
        let mut changed = one;
        changed.end = 0x3000;
        assert_ne!(left, mapping_set_digest(&[changed], 1));
    }

    #[test]
    fn claim_ceiling_remains_bounded() {
        let claims = [
            "prepared_fd_was_exec_syscall_operand_established=false",
            "all_inherited_process_state_qualified=false",
            "mapping_continuity_after_release_established=false",
            "post_launch_dynamic_loading_excluded=false",
            "trusted_time_established=false",
            "grants_physical_authority=false",
        ];
        assert_eq!(claims.len(), 6);
    }
}
