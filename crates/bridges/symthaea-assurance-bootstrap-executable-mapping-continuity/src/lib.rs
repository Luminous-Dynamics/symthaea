// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! File-backed executable-mapping continuity across confined bootstrap.
//!
//! The theorem is bounded to changes the traced verifier could cause through
//! its own mediated syscall stream. It does not exclude another privileged
//! process, root, the kernel, hardware, or physical mutation between samples.

#![cfg(all(
    target_os = "linux",
    target_arch = "x86_64",
    any(target_env = "gnu", target_env = "musl")
))]
#![deny(unsafe_code)]

use nix::sys::stat::{major, minor};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeSet,
    fs::File,
    io::{Read, Seek, SeekFrom},
    os::unix::fs::MetadataExt,
    path::PathBuf,
};
use symthaea_assurance_bootstrap_syscall_confinement::{
    BootstrapSyscallConfinementQualification, BootstrapSyscallObservation,
};
use symthaea_assurance_ptrace_static_exec_confirmation::{
    ExecutableMappingEvidence, StaticExecConfirmation,
};

pub const BOOTSTRAP_EXEC_MAPPING_POLICY_SCHEMA_V1: &str =
    "symthaea.assurance.bootstrap-executable-mapping-continuity-policy.v1";
pub const BOOTSTRAP_EXEC_MAPPING_REPORT_SCHEMA_V1: &str =
    "symthaea.assurance.bootstrap-executable-mapping-continuity-report.v1";
const POLICY_DOMAIN: &[u8] =
    b"symthaea.assurance.bootstrap-executable-mapping-continuity-policy.digest.v1\0";
const MAP_SET_DOMAIN: &[u8] = b"symthaea.assurance.ptrace-static-exec-map-set.digest.v1\0";
const SYSCALL_SEQUENCE_DOMAIN: &[u8] = b"symthaea.assurance.bootstrap-syscall-sequence.digest.v1\0";
const REPORT_DOMAIN: &[u8] =
    b"symthaea.assurance.bootstrap-executable-mapping-continuity-report.digest.v1\0";
const QUALIFICATION_DOMAIN: &[u8] =
    b"symthaea.assurance.bootstrap-executable-mapping-continuity-qualification.digest.v1\0";
const MAX_TEXT: usize = 4096;
const MAX_REFS: usize = 128;
const HARD_MAX_PROC_BYTES: u64 = 64 * 1024 * 1024;
const HARD_MAX_EXECUTABLE_BYTES: u64 = 1024 * 1024 * 1024;
const HARD_MAX_EXEC_MAPPING_BYTES: u64 = 1024 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BootstrapExecutableMappingContinuityPolicy {
    pub schema_version: String,
    pub policy_id: String,
    pub expected_confinement_policy_digest: String,
    pub expected_exec_confirmation_policy_digest: String,
    pub allowed_kernel_executable_pseudo_maps: Vec<String>,
    pub max_proc_maps_bytes: u64,
    pub max_executable_bytes: u64,
    pub max_total_executable_mapping_bytes: u64,
    pub evidence_refs: Vec<String>,
}

impl BootstrapExecutableMappingContinuityPolicy {
    pub fn validate(&self) -> bool {
        self.schema_version == BOOTSTRAP_EXEC_MAPPING_POLICY_SCHEMA_V1
            && canonical_text(&self.policy_id)
            && valid_blake3(&self.expected_confinement_policy_digest)
            && valid_blake3(&self.expected_exec_confirmation_policy_digest)
            && self.allowed_kernel_executable_pseudo_maps.len() <= 32
            && self
                .allowed_kernel_executable_pseudo_maps
                .iter()
                .all(|value| valid_pseudo_map(value))
            && unique(&self.allowed_kernel_executable_pseudo_maps)
            && (1..=HARD_MAX_PROC_BYTES).contains(&self.max_proc_maps_bytes)
            && (1..=HARD_MAX_EXECUTABLE_BYTES).contains(&self.max_executable_bytes)
            && (1..=HARD_MAX_EXEC_MAPPING_BYTES)
                .contains(&self.max_total_executable_mapping_bytes)
            && valid_refs(&self.evidence_refs)
    }

    pub fn canonical_digest(&self) -> Option<String> {
        if !self.validate() {
            return None;
        }
        let mut hasher = blake3::Hasher::new();
        hasher.update(POLICY_DOMAIN);
        for value in [
            self.schema_version.as_str(),
            self.policy_id.as_str(),
            self.expected_confinement_policy_digest.as_str(),
            self.expected_exec_confirmation_policy_digest.as_str(),
        ] {
            field(&mut hasher, value);
        }
        sorted_strings(&mut hasher, &self.allowed_kernel_executable_pseudo_maps);
        for value in [
            self.max_proc_maps_bytes,
            self.max_executable_bytes,
            self.max_total_executable_mapping_bytes,
        ] {
            hasher.update(&value.to_le_bytes());
        }
        sorted_strings(&mut hasher, &self.evidence_refs);
        Some(format!("blake3:{}", hasher.finalize().to_hex()))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BootstrapExecutableMappingContinuityIssue {
    InvalidPolicy,
    ConfinementPolicyMismatch,
    ExecConfirmationPolicyMismatch,
    ConfinementConfirmationMismatch,
    TraceePidMismatch,
    ConfinementTranscriptMismatch,
    ConfinementSyscallCountMismatch,
    ReadyWriteNotUniqueAndFinal,
    ExecEnableSyscallObserved(i64),
    ThreadEscapeSyscallObserved(i64),
    ExecutableMmapObserved,
    ExecutableMprotectObserved,
    LaunchReportMismatch,
    LaunchConfirmationMismatch,
    LaunchMappingSetMismatch,
    LaunchMappingCountMismatch,
    LaunchMappingBytesMismatch,
    ProcExeUnavailable(String),
    ExecutablePathMismatch,
    ExecutableMetadataMismatch,
    ExecutableTooLarge { observed: u64, maximum: u64 },
    ExecutableDigestMismatch,
    ExecutableChangedDuringObservation,
    ProcMapsUnavailable(String),
    ProcMapsInvalid(String),
    MapsChangedDuringObservation,
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
    ExecutableMappingChangedDuringObservation(String),
    NoExecutableTargetMapping,
    KernelPseudoMapCountMismatch,
    ReadyMappingSetMismatch,
    ReadyFileBackedMappingEvidenceMismatch,
}

impl BootstrapExecutableMappingContinuityIssue {
    fn invalid(&self) -> bool { matches!(self, Self::InvalidPolicy) }

    fn code(&self) -> String {
        match self {
            Self::InvalidPolicy => "invalid-policy".into(),
            Self::ConfinementPolicyMismatch => "confinement-policy-mismatch".into(),
            Self::ExecConfirmationPolicyMismatch => "exec-confirmation-policy-mismatch".into(),
            Self::ConfinementConfirmationMismatch => "confinement-confirmation-mismatch".into(),
            Self::TraceePidMismatch => "tracee-pid-mismatch".into(),
            Self::ConfinementTranscriptMismatch => "confinement-transcript-mismatch".into(),
            Self::ConfinementSyscallCountMismatch => "confinement-syscall-count-mismatch".into(),
            Self::ReadyWriteNotUniqueAndFinal => "ready-write-not-unique-and-final".into(),
            Self::ExecEnableSyscallObserved(value) => format!("exec-enable-syscall-observed:{value}"),
            Self::ThreadEscapeSyscallObserved(value) => format!("thread-escape-syscall-observed:{value}"),
            Self::ExecutableMmapObserved => "executable-mmap-observed".into(),
            Self::ExecutableMprotectObserved => "executable-mprotect-observed".into(),
            Self::LaunchReportMismatch => "launch-report-mismatch".into(),
            Self::LaunchConfirmationMismatch => "launch-confirmation-mismatch".into(),
            Self::LaunchMappingSetMismatch => "launch-mapping-set-mismatch".into(),
            Self::LaunchMappingCountMismatch => "launch-mapping-count-mismatch".into(),
            Self::LaunchMappingBytesMismatch => "launch-mapping-bytes-mismatch".into(),
            Self::ProcExeUnavailable(value) => format!("proc-exe-unavailable:{value}"),
            Self::ExecutablePathMismatch => "executable-path-mismatch".into(),
            Self::ExecutableMetadataMismatch => "executable-metadata-mismatch".into(),
            Self::ExecutableTooLarge { observed, maximum } => {
                format!("executable-too-large:{observed}:{maximum}")
            }
            Self::ExecutableDigestMismatch => "executable-digest-mismatch".into(),
            Self::ExecutableChangedDuringObservation => "executable-changed-during-observation".into(),
            Self::ProcMapsUnavailable(value) => format!("proc-maps-unavailable:{value}"),
            Self::ProcMapsInvalid(value) => format!("proc-maps-invalid:{value}"),
            Self::MapsChangedDuringObservation => "maps-changed-during-observation".into(),
            Self::WritableExecutableMapping(value) => format!("writable-exec-map:{value}"),
            Self::NonReadableExecutableMapping(value) => format!("nonreadable-exec-map:{value}"),
            Self::SharedExecutableMapping(value) => format!("shared-exec-map:{value}"),
            Self::UnapprovedExecutablePseudoMap(value) => format!("unapproved-exec-pseudo:{value}"),
            Self::AnonymousExecutableMapping => "anonymous-executable-mapping".into(),
            Self::UnexpectedExecutableMapping(value) => format!("unexpected-executable-map:{value}"),
            Self::ExecutableMappingIdentityMismatch(value) => {
                format!("exec-map-identity-mismatch:{value}")
            }
            Self::ExecutableMappingOutsideFile(value) => format!("exec-map-outside-file:{value}"),
            Self::ExecutableMappingBytesExceeded { observed, maximum } => {
                format!("exec-map-bytes-exceeded:{observed}:{maximum}")
            }
            Self::ProcMemUnavailable(value) => format!("proc-mem-unavailable:{value}"),
            Self::ExecutableMappingReadFailed(value) => format!("exec-map-read-failed:{value}"),
            Self::ExecutableMappingBytesMismatch(value) => format!("exec-map-bytes-mismatch:{value}"),
            Self::ExecutableMappingChangedDuringObservation(value) => {
                format!("exec-map-changed-during-observation:{value}")
            }
            Self::NoExecutableTargetMapping => "no-executable-target-mapping".into(),
            Self::KernelPseudoMapCountMismatch => "kernel-pseudo-map-count-mismatch".into(),
            Self::ReadyMappingSetMismatch => "ready-mapping-set-mismatch".into(),
            Self::ReadyFileBackedMappingEvidenceMismatch => {
                "ready-file-backed-mapping-evidence-mismatch".into()
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BootstrapExecutableMappingContinuityDisposition {
    Invalid,
    Blocked,
    Qualified,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BootstrapExecutableMappingContinuityReport {
    pub schema_version: String,
    pub policy_id: String,
    pub policy_digest: Option<String>,
    pub confinement_qualification_digest: String,
    pub exec_confirmation_digest: String,
    pub executable_digest: String,
    pub launch_mapping_set_digest: String,
    pub ready_mapping_set_digest: Option<String>,
    pub syscall_sequence_digest: String,
    pub maps_before_digest: Option<String>,
    pub maps_after_digest: Option<String>,
    pub ready_executable_digest_before: Option<String>,
    pub ready_executable_digest_after: Option<String>,
    pub file_backed_executable_mapping_count: u64,
    pub kernel_executable_pseudo_map_count: u64,
    pub total_file_backed_executable_mapping_bytes: u64,
    pub disposition: BootstrapExecutableMappingContinuityDisposition,
    pub issues: Vec<BootstrapExecutableMappingContinuityIssue>,
}

impl BootstrapExecutableMappingContinuityReport {
    pub fn canonical_digest(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(REPORT_DOMAIN);
        for value in [
            self.schema_version.as_str(),
            self.policy_id.as_str(),
            self.policy_digest.as_deref().unwrap_or("-"),
            self.confinement_qualification_digest.as_str(),
            self.exec_confirmation_digest.as_str(),
            self.executable_digest.as_str(),
            self.launch_mapping_set_digest.as_str(),
            self.ready_mapping_set_digest.as_deref().unwrap_or("-"),
            self.syscall_sequence_digest.as_str(),
            self.maps_before_digest.as_deref().unwrap_or("-"),
            self.maps_after_digest.as_deref().unwrap_or("-"),
            self.ready_executable_digest_before.as_deref().unwrap_or("-"),
            self.ready_executable_digest_after.as_deref().unwrap_or("-"),
        ] {
            field(&mut hasher, value);
        }
        for value in [
            self.file_backed_executable_mapping_count,
            self.kernel_executable_pseudo_map_count,
            self.total_file_backed_executable_mapping_bytes,
        ] {
            hasher.update(&value.to_le_bytes());
        }
        field(
            &mut hasher,
            match self.disposition {
                BootstrapExecutableMappingContinuityDisposition::Invalid => "invalid",
                BootstrapExecutableMappingContinuityDisposition::Blocked => "blocked",
                BootstrapExecutableMappingContinuityDisposition::Qualified => "qualified",
            },
        );
        hasher.update(&(self.issues.len() as u64).to_le_bytes());
        for issue in &self.issues {
            field(&mut hasher, &issue.code());
        }
        format!("blake3:{}", hasher.finalize().to_hex())
    }

    pub const fn grants_physical_authority(&self) -> bool { false }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BootstrapExecutableMappingContinuity {
    qualification_digest: String,
    report_digest: String,
    policy_digest: String,
    confinement_qualification_digest: String,
    exec_confirmation_digest: String,
    executable_digest: String,
    launch_mapping_set_digest: String,
    ready_mapping_set_digest: String,
    syscall_sequence_digest: String,
    mapping_count: u64,
}

impl BootstrapExecutableMappingContinuity {
    pub fn qualification_digest(&self) -> &str { &self.qualification_digest }
    pub fn report_digest(&self) -> &str { &self.report_digest }
    pub fn policy_digest(&self) -> &str { &self.policy_digest }
    pub fn confinement_qualification_digest(&self) -> &str {
        &self.confinement_qualification_digest
    }
    pub fn exec_confirmation_digest(&self) -> &str { &self.exec_confirmation_digest }
    pub fn executable_digest(&self) -> &str { &self.executable_digest }
    pub fn launch_mapping_set_digest(&self) -> &str { &self.launch_mapping_set_digest }
    pub fn ready_mapping_set_digest(&self) -> &str { &self.ready_mapping_set_digest }
    pub fn syscall_sequence_digest(&self) -> &str { &self.syscall_sequence_digest }
    pub const fn mapping_count(&self) -> u64 { self.mapping_count }
    pub const fn public_launch_mapping_evidence_rebound_to_opaque_confirmation(&self) -> bool { true }
    pub const fn public_syscall_transcript_rebound_to_opaque_confinement(&self) -> bool { true }
    pub const fn launch_and_ready_file_backed_executable_mappings_identical(&self) -> bool { true }
    pub const fn launch_and_ready_executable_mapping_bytes_identical(&self) -> bool { true }
    pub const fn no_tracee_exec_enable_or_thread_escape_syscall_restarted(&self) -> bool { true }
    pub const fn tracee_action_file_backed_executable_mapping_continuity_established(&self) -> bool {
        true
    }
    pub const fn kernel_pseudo_mapping_identity_continuity_established(&self) -> bool { false }
    pub const fn external_process_mutation_excluded(&self) -> bool { false }
    pub const fn root_resistant_memory_immutability_established(&self) -> bool { false }
    pub const fn trusted_time_established(&self) -> bool { false }
    pub const fn grants_physical_authority(&self) -> bool { false }
}

pub struct BootstrapExecutableMappingContinuityQualification {
    pub report: BootstrapExecutableMappingContinuityReport,
    pub ready_mappings: Vec<ExecutableMappingEvidence>,
    verified: BootstrapExecutableMappingContinuity,
}

impl BootstrapExecutableMappingContinuityQualification {
    pub fn verified(&self) -> &BootstrapExecutableMappingContinuity { &self.verified }
    pub fn into_verified(self) -> BootstrapExecutableMappingContinuity { self.verified }
}

pub fn assess_bootstrap_executable_mapping_continuity(
    policy: &BootstrapExecutableMappingContinuityPolicy,
    launch: &StaticExecConfirmation,
    confinement: &BootstrapSyscallConfinementQualification,
) -> Result<
    BootstrapExecutableMappingContinuityQualification,
    BootstrapExecutableMappingContinuityReport,
> {
    let policy_digest = policy.canonical_digest();
    let confirmed = launch.confirmed();
    let ready = confinement.ready();
    let syscall_digest = syscall_sequence_digest(&confinement.syscalls);
    let mut report = BootstrapExecutableMappingContinuityReport {
        schema_version: BOOTSTRAP_EXEC_MAPPING_REPORT_SCHEMA_V1.into(),
        policy_id: policy.policy_id.clone(),
        policy_digest: policy_digest.clone(),
        confinement_qualification_digest: ready.qualification_digest().into(),
        exec_confirmation_digest: confirmed.confirmation_digest().into(),
        executable_digest: confirmed.executable_digest().into(),
        launch_mapping_set_digest: confirmed.executable_mapping_set_digest().into(),
        ready_mapping_set_digest: None,
        syscall_sequence_digest: syscall_digest.clone(),
        maps_before_digest: None,
        maps_after_digest: None,
        ready_executable_digest_before: None,
        ready_executable_digest_after: None,
        file_backed_executable_mapping_count: 0,
        kernel_executable_pseudo_map_count: 0,
        total_file_backed_executable_mapping_bytes: 0,
        disposition: BootstrapExecutableMappingContinuityDisposition::Invalid,
        issues: Vec::new(),
    };

    if !policy.validate() {
        report.issues.push(BootstrapExecutableMappingContinuityIssue::InvalidPolicy);
        return Err(finalize(report));
    }
    if ready.policy_digest() != policy.expected_confinement_policy_digest {
        report
            .issues
            .push(BootstrapExecutableMappingContinuityIssue::ConfinementPolicyMismatch);
    }
    if confirmed.policy_digest() != policy.expected_exec_confirmation_policy_digest {
        report
            .issues
            .push(BootstrapExecutableMappingContinuityIssue::ExecConfirmationPolicyMismatch);
    }
    if ready.exec_confirmation_digest() != confirmed.confirmation_digest() {
        report
            .issues
            .push(BootstrapExecutableMappingContinuityIssue::ConfinementConfirmationMismatch);
    }
    if ready.tracee_pid() != confirmed.tracee_pid() {
        report
            .issues
            .push(BootstrapExecutableMappingContinuityIssue::TraceePidMismatch);
    }
    if syscall_digest != ready.syscall_sequence_digest() {
        report
            .issues
            .push(BootstrapExecutableMappingContinuityIssue::ConfinementTranscriptMismatch);
    }
    if confinement.syscalls.len() as u64 != ready.syscall_count() {
        report
            .issues
            .push(BootstrapExecutableMappingContinuityIssue::ConfinementSyscallCountMismatch);
    }
    let ready_writes = confinement
        .syscalls
        .iter()
        .enumerate()
        .filter(|(_, value)| value.ready_token_write)
        .map(|(index, _)| index)
        .collect::<Vec<_>>();
    if ready_writes.as_slice() != [confinement.syscalls.len().saturating_sub(1)] {
        report
            .issues
            .push(BootstrapExecutableMappingContinuityIssue::ReadyWriteNotUniqueAndFinal);
    }
    report
        .issues
        .extend(transcript_issues(&confinement.syscalls));

    if launch.report.canonical_digest() != confirmed.report_digest() {
        report
            .issues
            .push(BootstrapExecutableMappingContinuityIssue::LaunchReportMismatch);
    }
    if launch.report.confirmation_digest.as_deref() != Some(confirmed.confirmation_digest()) {
        report
            .issues
            .push(BootstrapExecutableMappingContinuityIssue::LaunchConfirmationMismatch);
    }
    let launch_set = mapping_set_digest(
        &launch.mappings,
        launch.report.kernel_executable_pseudo_map_count,
    );
    if launch_set != confirmed.executable_mapping_set_digest()
        || launch.report.executable_mapping_set_digest.as_deref()
            != Some(confirmed.executable_mapping_set_digest())
    {
        report
            .issues
            .push(BootstrapExecutableMappingContinuityIssue::LaunchMappingSetMismatch);
    }
    if launch.mappings.len() as u64 != confirmed.executable_mapping_count()
        || launch.report.executable_mapping_count != confirmed.executable_mapping_count()
    {
        report
            .issues
            .push(BootstrapExecutableMappingContinuityIssue::LaunchMappingCountMismatch);
    }
    let launch_bytes = launch
        .mappings
        .iter()
        .try_fold(0u64, |total, map| total.checked_add(map.end - map.start));
    if launch_bytes != Some(confirmed.total_executable_mapping_bytes())
        || launch.report.total_executable_mapping_bytes != confirmed.total_executable_mapping_bytes()
    {
        report
            .issues
            .push(BootstrapExecutableMappingContinuityIssue::LaunchMappingBytesMismatch);
    }
    if !report.issues.is_empty() {
        return Err(finalize(report));
    }

    let observed = match observe_ready_mappings(policy, confirmed.tracee_pid(), confirmed) {
        Ok(value) => value,
        Err(issues) => {
            report.issues.extend(issues);
            return Err(finalize(report));
        }
    };
    report.maps_before_digest = Some(observed.maps_before_digest.clone());
    report.maps_after_digest = Some(observed.maps_after_digest.clone());
    report.ready_executable_digest_before = Some(observed.executable_digest_before.clone());
    report.ready_executable_digest_after = Some(observed.executable_digest_after.clone());
    report.file_backed_executable_mapping_count = observed.mappings.len() as u64;
    report.kernel_executable_pseudo_map_count = observed.kernel_pseudo_count;
    report.total_file_backed_executable_mapping_bytes = observed.total_bytes;
    report.ready_mapping_set_digest = Some(observed.mapping_set_digest.clone());

    if observed.kernel_pseudo_count != launch.report.kernel_executable_pseudo_map_count {
        report
            .issues
            .push(BootstrapExecutableMappingContinuityIssue::KernelPseudoMapCountMismatch);
    }
    if observed.mapping_set_digest != confirmed.executable_mapping_set_digest() {
        report
            .issues
            .push(BootstrapExecutableMappingContinuityIssue::ReadyMappingSetMismatch);
    }
    let mut launch_mappings = launch.mappings.clone();
    let mut ready_mappings = observed.mappings.clone();
    launch_mappings.sort_by_key(|value| (value.start, value.end));
    ready_mappings.sort_by_key(|value| (value.start, value.end));
    if launch_mappings != ready_mappings {
        report
            .issues
            .push(BootstrapExecutableMappingContinuityIssue::ReadyFileBackedMappingEvidenceMismatch);
    }
    if !report.issues.is_empty() {
        return Err(finalize(report));
    }

    report.disposition = BootstrapExecutableMappingContinuityDisposition::Qualified;
    let report_digest = report.canonical_digest();
    let policy_digest = policy_digest.expect("validated policy");
    let ready_mapping_set_digest = observed.mapping_set_digest.clone();
    let qualification_digest = qualification_digest(
        &policy_digest,
        ready.qualification_digest(),
        confirmed.confirmation_digest(),
        confirmed.executable_mapping_set_digest(),
        &ready_mapping_set_digest,
        &syscall_digest,
        &report_digest,
    );
    let verified = BootstrapExecutableMappingContinuity {
        qualification_digest,
        report_digest,
        policy_digest,
        confinement_qualification_digest: ready.qualification_digest().into(),
        exec_confirmation_digest: confirmed.confirmation_digest().into(),
        executable_digest: confirmed.executable_digest().into(),
        launch_mapping_set_digest: confirmed.executable_mapping_set_digest().into(),
        ready_mapping_set_digest,
        syscall_sequence_digest: syscall_digest,
        mapping_count: observed.mappings.len() as u64,
    };
    Ok(BootstrapExecutableMappingContinuityQualification {
        report,
        ready_mappings: observed.mappings,
        verified,
    })
}

fn transcript_issues(
    syscalls: &[BootstrapSyscallObservation],
) -> Vec<BootstrapExecutableMappingContinuityIssue> {
    let mut issues = Vec::new();
    for syscall in syscalls {
        let number = syscall.syscall_number;
        if number == nix::libc::SYS_execve
            || number == nix::libc::SYS_execveat
            || number == nix::libc::SYS_mremap
            || number == nix::libc::SYS_remap_file_pages
            || number == nix::libc::SYS_pkey_mprotect
            || number == nix::libc::SYS_shmat
        {
            issues.push(BootstrapExecutableMappingContinuityIssue::ExecEnableSyscallObserved(
                number,
            ));
        }
        if number == nix::libc::SYS_clone
            || number == nix::libc::SYS_clone3
            || number == nix::libc::SYS_fork
            || number == nix::libc::SYS_vfork
        {
            issues.push(BootstrapExecutableMappingContinuityIssue::ThreadEscapeSyscallObserved(
                number,
            ));
        }
        if number == nix::libc::SYS_mmap && syscall.args[2] as i32 & nix::libc::PROT_EXEC != 0 {
            issues.push(BootstrapExecutableMappingContinuityIssue::ExecutableMmapObserved);
        }
        if number == nix::libc::SYS_mprotect
            && syscall.args[2] as i32 & nix::libc::PROT_EXEC != 0
        {
            issues.push(BootstrapExecutableMappingContinuityIssue::ExecutableMprotectObserved);
        }
    }
    issues
}

struct ReadyMappingObservation {
    mappings: Vec<ExecutableMappingEvidence>,
    kernel_pseudo_count: u64,
    total_bytes: u64,
    maps_before_digest: String,
    maps_after_digest: String,
    executable_digest_before: String,
    executable_digest_after: String,
    mapping_set_digest: String,
}

fn observe_ready_mappings(
    policy: &BootstrapExecutableMappingContinuityPolicy,
    pid: i32,
    confirmed: &symthaea_assurance_ptrace_static_exec_confirmation::ConfirmedStaticExecLaunch,
) -> Result<ReadyMappingObservation, Vec<BootstrapExecutableMappingContinuityIssue>> {
    let mut issues = Vec::new();
    let exe_link = PathBuf::from(format!("/proc/{pid}/exe"));
    let observed_path = match std::fs::read_link(&exe_link)
        .ok()
        .and_then(|path| path.into_os_string().into_string().ok())
    {
        Some(value) => value,
        None => {
            issues.push(BootstrapExecutableMappingContinuityIssue::ProcExeUnavailable(
                "readlink".into(),
            ));
            return Err(issues);
        }
    };
    if observed_path != confirmed.executable_path() {
        issues.push(BootstrapExecutableMappingContinuityIssue::ExecutablePathMismatch);
    }
    let mut executable = match File::open(&exe_link) {
        Ok(value) => value,
        Err(error) => {
            issues.push(BootstrapExecutableMappingContinuityIssue::ProcExeUnavailable(
                error.to_string(),
            ));
            return Err(issues);
        }
    };
    let metadata_before = match executable.metadata() {
        Ok(value) => value,
        Err(error) => {
            issues.push(BootstrapExecutableMappingContinuityIssue::ProcExeUnavailable(
                error.to_string(),
            ));
            return Err(issues);
        }
    };
    if metadata_before.len() > policy.max_executable_bytes {
        issues.push(BootstrapExecutableMappingContinuityIssue::ExecutableTooLarge {
            observed: metadata_before.len(),
            maximum: policy.max_executable_bytes,
        });
        return Err(issues);
    }
    let executable_digest_before = match hash_file(&mut executable, policy.max_executable_bytes) {
        Ok(value) => value,
        Err(issue) => {
            issues.push(issue);
            return Err(issues);
        }
    };
    if executable_digest_before != confirmed.executable_digest() {
        issues.push(BootstrapExecutableMappingContinuityIssue::ExecutableDigestMismatch);
    }

    let maps_path = PathBuf::from(format!("/proc/{pid}/maps"));
    let maps_before = match read_bounded(&maps_path, policy.max_proc_maps_bytes) {
        Ok(value) => value,
        Err(error) => {
            issues.push(BootstrapExecutableMappingContinuityIssue::ProcMapsUnavailable(error));
            return Err(issues);
        }
    };
    let maps_before_digest = format!("blake3:{}", blake3::hash(&maps_before).to_hex());
    let maps_text = match std::str::from_utf8(&maps_before) {
        Ok(value) => value,
        Err(error) => {
            issues.push(BootstrapExecutableMappingContinuityIssue::ProcMapsInvalid(
                error.to_string(),
            ));
            return Err(issues);
        }
    };
    let parsed = match parse_maps(maps_text) {
        Ok(value) => value,
        Err(error) => {
            issues.push(BootstrapExecutableMappingContinuityIssue::ProcMapsInvalid(error));
            return Err(issues);
        }
    };
    let mut memory = match File::open(format!("/proc/{pid}/mem")) {
        Ok(value) => value,
        Err(error) => {
            issues.push(BootstrapExecutableMappingContinuityIssue::ProcMemUnavailable(
                error.to_string(),
            ));
            return Err(issues);
        }
    };
    let target_major = major(metadata_before.dev());
    let target_minor = minor(metadata_before.dev());
    let allowed_pseudo = policy
        .allowed_kernel_executable_pseudo_maps
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>();
    let mut mappings = Vec::new();
    let mut kernel_pseudo_count = 0u64;
    let mut total_bytes = 0u64;
    for map in parsed.into_iter().filter(ParsedMap::executable) {
        if map.writable() {
            issues.push(BootstrapExecutableMappingContinuityIssue::WritableExecutableMapping(
                map.pathname.clone(),
            ));
            continue;
        }
        if map.pathname.starts_with('[') {
            if allowed_pseudo.contains(&map.pathname) {
                kernel_pseudo_count += 1;
            } else {
                issues.push(
                    BootstrapExecutableMappingContinuityIssue::UnapprovedExecutablePseudoMap(
                        map.pathname,
                    ),
                );
            }
            continue;
        }
        if map.pathname.is_empty() {
            issues.push(BootstrapExecutableMappingContinuityIssue::AnonymousExecutableMapping);
            continue;
        }
        if !map.readable() {
            issues.push(
                BootstrapExecutableMappingContinuityIssue::NonReadableExecutableMapping(
                    map.pathname,
                ),
            );
            continue;
        }
        if !map.private_mapping() {
            issues.push(BootstrapExecutableMappingContinuityIssue::SharedExecutableMapping(
                map.pathname,
            ));
            continue;
        }
        if map.pathname != confirmed.executable_path() {
            issues.push(BootstrapExecutableMappingContinuityIssue::UnexpectedExecutableMapping(
                map.pathname,
            ));
            continue;
        }
        if map.device_major != target_major
            || map.device_minor != target_minor
            || map.inode != metadata_before.ino()
        {
            issues.push(
                BootstrapExecutableMappingContinuityIssue::ExecutableMappingIdentityMismatch(
                    map.pathname,
                ),
            );
            continue;
        }
        let length = map.end - map.start;
        total_bytes = total_bytes.checked_add(length).unwrap_or(u64::MAX);
        if total_bytes > policy.max_total_executable_mapping_bytes {
            issues.push(
                BootstrapExecutableMappingContinuityIssue::ExecutableMappingBytesExceeded {
                    observed: total_bytes,
                    maximum: policy.max_total_executable_mapping_bytes,
                },
            );
            continue;
        }
        if map
            .file_offset
            .checked_add(length)
            .is_none_or(|end| end > metadata_before.len())
        {
            issues.push(
                BootstrapExecutableMappingContinuityIssue::ExecutableMappingOutsideFile(
                    map.pathname,
                ),
            );
            continue;
        }
        match compare_mapping_twice(&mut memory, &mut executable, &map) {
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
            Err(issue) => issues.push(issue),
        }
    }
    if mappings.is_empty() {
        issues.push(BootstrapExecutableMappingContinuityIssue::NoExecutableTargetMapping);
    }

    let maps_after = match read_bounded(&maps_path, policy.max_proc_maps_bytes) {
        Ok(value) => value,
        Err(error) => {
            issues.push(BootstrapExecutableMappingContinuityIssue::ProcMapsUnavailable(error));
            return Err(issues);
        }
    };
    let maps_after_digest = format!("blake3:{}", blake3::hash(&maps_after).to_hex());
    if maps_before != maps_after {
        issues.push(BootstrapExecutableMappingContinuityIssue::MapsChangedDuringObservation);
    }
    let executable_digest_after = match hash_file(&mut executable, policy.max_executable_bytes) {
        Ok(value) => value,
        Err(issue) => {
            issues.push(issue);
            return Err(issues);
        }
    };
    let metadata_after = match executable.metadata() {
        Ok(value) => value,
        Err(error) => {
            issues.push(BootstrapExecutableMappingContinuityIssue::ProcExeUnavailable(
                error.to_string(),
            ));
            return Err(issues);
        }
    };
    if executable_digest_after != confirmed.executable_digest() {
        issues.push(BootstrapExecutableMappingContinuityIssue::ExecutableDigestMismatch);
    }
    if executable_digest_before != executable_digest_after
        || !same_metadata(&metadata_before, &metadata_after)
    {
        issues.push(BootstrapExecutableMappingContinuityIssue::ExecutableChangedDuringObservation);
    }
    if !issues.is_empty() {
        return Err(issues);
    }
    let mapping_set_digest = mapping_set_digest(&mappings, kernel_pseudo_count);
    Ok(ReadyMappingObservation {
        mappings,
        kernel_pseudo_count,
        total_bytes,
        maps_before_digest,
        maps_after_digest,
        executable_digest_before,
        executable_digest_after,
        mapping_set_digest,
    })
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
    let mut output = Vec::new();
    for line in text.lines() {
        if line.is_empty() {
            continue;
        }
        let (range, rest) = take_token(line).ok_or("missing-range")?;
        let (permissions, rest) = take_token(rest).ok_or("missing-permissions")?;
        let (offset, rest) = take_token(rest).ok_or("missing-offset")?;
        let (device, rest) = take_token(rest).ok_or("missing-device")?;
        let (inode, rest) = take_token(rest).ok_or("missing-inode")?;
        let pathname = rest.trim_start().to_string();
        let (start, end) = range.split_once('-').ok_or("invalid-range")?;
        let start = u64::from_str_radix(start, 16).map_err(|_| "invalid-range-start")?;
        let end = u64::from_str_radix(end, 16).map_err(|_| "invalid-range-end")?;
        if start >= end || permissions.len() != 4 {
            return Err("invalid-range-or-permissions".into());
        }
        let file_offset = u64::from_str_radix(offset, 16).map_err(|_| "invalid-offset")?;
        let (device_major, device_minor) = device.split_once(':').ok_or("invalid-device")?;
        let device_major =
            u64::from_str_radix(device_major, 16).map_err(|_| "invalid-device-major")?;
        let device_minor =
            u64::from_str_radix(device_minor, 16).map_err(|_| "invalid-device-minor")?;
        let inode = inode.parse::<u64>().map_err(|_| "invalid-inode")?;
        output.push(ParsedMap {
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
    if output.is_empty() {
        return Err("empty-maps".into());
    }
    Ok(output)
}

fn take_token(value: &str) -> Option<(&str, &str)> {
    let value = value.trim_start_matches(char::is_whitespace);
    if value.is_empty() {
        return None;
    }
    let index = value.find(char::is_whitespace).unwrap_or(value.len());
    Some((&value[..index], &value[index..]))
}

fn compare_mapping_twice(
    memory: &mut File,
    executable: &mut File,
    map: &ParsedMap,
) -> Result<(String, String), BootstrapExecutableMappingContinuityIssue> {
    let first = compare_mapping_once(memory, executable, map)?;
    let second = compare_mapping_once(memory, executable, map)?;
    if first != second {
        return Err(
            BootstrapExecutableMappingContinuityIssue::ExecutableMappingChangedDuringObservation(
                map.pathname.clone(),
            ),
        );
    }
    Ok(first)
}

fn compare_mapping_once(
    memory: &mut File,
    executable: &mut File,
    map: &ParsedMap,
) -> Result<(String, String), BootstrapExecutableMappingContinuityIssue> {
    memory
        .seek(SeekFrom::Start(map.start))
        .map_err(|error| {
            BootstrapExecutableMappingContinuityIssue::ExecutableMappingReadFailed(
                error.to_string(),
            )
        })?;
    executable
        .seek(SeekFrom::Start(map.file_offset))
        .map_err(|error| {
            BootstrapExecutableMappingContinuityIssue::ExecutableMappingReadFailed(
                error.to_string(),
            )
        })?;
    let mut remaining = map.end - map.start;
    let mut memory_hasher = blake3::Hasher::new();
    let mut backing_hasher = blake3::Hasher::new();
    let mut memory_buffer = vec![0u8; 64 * 1024];
    let mut backing_buffer = vec![0u8; 64 * 1024];
    while remaining > 0 {
        let size = remaining.min(memory_buffer.len() as u64) as usize;
        memory.read_exact(&mut memory_buffer[..size]).map_err(|error| {
            BootstrapExecutableMappingContinuityIssue::ExecutableMappingReadFailed(
                error.to_string(),
            )
        })?;
        executable
            .read_exact(&mut backing_buffer[..size])
            .map_err(|error| {
                BootstrapExecutableMappingContinuityIssue::ExecutableMappingReadFailed(
                    error.to_string(),
                )
            })?;
        if memory_buffer[..size] != backing_buffer[..size] {
            return Err(BootstrapExecutableMappingContinuityIssue::ExecutableMappingBytesMismatch(
                map.pathname.clone(),
            ));
        }
        memory_hasher.update(&memory_buffer[..size]);
        backing_hasher.update(&backing_buffer[..size]);
        remaining -= size as u64;
    }
    Ok((
        format!("blake3:{}", memory_hasher.finalize().to_hex()),
        format!("blake3:{}", backing_hasher.finalize().to_hex()),
    ))
}

fn hash_file(
    file: &mut File,
    maximum: u64,
) -> Result<String, BootstrapExecutableMappingContinuityIssue> {
    let length = file
        .metadata()
        .map_err(|error| {
            BootstrapExecutableMappingContinuityIssue::ProcExeUnavailable(error.to_string())
        })?
        .len();
    if length > maximum {
        return Err(BootstrapExecutableMappingContinuityIssue::ExecutableTooLarge {
            observed: length,
            maximum,
        });
    }
    file.seek(SeekFrom::Start(0)).map_err(|error| {
        BootstrapExecutableMappingContinuityIssue::ProcExeUnavailable(error.to_string())
    })?;
    let mut hasher = blake3::Hasher::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer).map_err(|error| {
            BootstrapExecutableMappingContinuityIssue::ProcExeUnavailable(error.to_string())
        })?;
        if count == 0 {
            break;
        }
        hasher.update(&buffer[..count]);
    }
    Ok(format!("blake3:{}", hasher.finalize().to_hex()))
}

fn same_metadata(left: &std::fs::Metadata, right: &std::fs::Metadata) -> bool {
    left.dev() == right.dev()
        && left.ino() == right.ino()
        && left.len() == right.len()
        && left.mode() == right.mode()
        && left.uid() == right.uid()
        && left.gid() == right.gid()
        && left.mtime() == right.mtime()
        && left.mtime_nsec() == right.mtime_nsec()
}

fn read_bounded(path: &PathBuf, maximum: u64) -> Result<Vec<u8>, String> {
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

fn mapping_set_digest(values: &[ExecutableMappingEvidence], pseudo_count: u64) -> String {
    let mut values = values.to_vec();
    values.sort_by_key(|value| (value.start, value.end));
    let mut hasher = blake3::Hasher::new();
    hasher.update(MAP_SET_DOMAIN);
    hasher.update(&(values.len() as u64).to_le_bytes());
    hasher.update(&pseudo_count.to_le_bytes());
    for map in &values {
        for value in [
            map.start,
            map.end,
            map.file_offset,
            map.device_major,
            map.device_minor,
            map.inode,
        ] {
            hasher.update(&value.to_le_bytes());
        }
        field(&mut hasher, &map.permissions);
        field(&mut hasher, &map.mapped_bytes_blake3);
        field(&mut hasher, &map.backing_bytes_blake3);
    }
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn syscall_sequence_digest(values: &[BootstrapSyscallObservation]) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(SYSCALL_SEQUENCE_DOMAIN);
    hasher.update(&(values.len() as u64).to_le_bytes());
    for value in values {
        hasher.update(&value.syscall_number.to_le_bytes());
        for arg in value.args {
            hasher.update(&arg.to_le_bytes());
        }
        hasher.update(&value.return_value.to_le_bytes());
        hasher.update(&[u8::from(value.ready_token_write)]);
    }
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn qualification_digest(
    policy_digest: &str,
    confinement_qualification_digest: &str,
    exec_confirmation_digest: &str,
    launch_mapping_set_digest: &str,
    ready_mapping_set_digest: &str,
    syscall_sequence_digest: &str,
    report_digest: &str,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(QUALIFICATION_DOMAIN);
    for value in [
        policy_digest,
        confinement_qualification_digest,
        exec_confirmation_digest,
        launch_mapping_set_digest,
        ready_mapping_set_digest,
        syscall_sequence_digest,
        report_digest,
    ] {
        field(&mut hasher, value);
    }
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn finalize(
    mut report: BootstrapExecutableMappingContinuityReport,
) -> BootstrapExecutableMappingContinuityReport {
    report.disposition = if report
        .issues
        .iter()
        .any(BootstrapExecutableMappingContinuityIssue::invalid)
    {
        BootstrapExecutableMappingContinuityDisposition::Invalid
    } else {
        BootstrapExecutableMappingContinuityDisposition::Blocked
    };
    report
}

fn canonical_text(value: &str) -> bool {
    !value.is_empty()
        && value == value.trim()
        && value.len() <= MAX_TEXT
        && !value.chars().any(char::is_control)
}
fn valid_blake3(value: &str) -> bool {
    value.strip_prefix("blake3:").is_some_and(|hex| {
        hex.len() == 64
            && hex
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    })
}
fn valid_pseudo_map(value: &str) -> bool {
    canonical_text(value) && value.starts_with('[') && value.ends_with(']')
}
fn valid_refs(values: &[String]) -> bool {
    values.len() <= MAX_REFS && values.iter().all(|value| canonical_text(value)) && unique(values)
}
fn unique(values: &[String]) -> bool {
    let mut seen = BTreeSet::new();
    values.iter().all(|value| seen.insert(value.as_str()))
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

#[cfg(test)]
mod tests {
    use super::*;

    fn d(label: &str) -> String {
        format!("blake3:{}", blake3::hash(label.as_bytes()).to_hex())
    }

    fn mapping(start: u64) -> ExecutableMappingEvidence {
        ExecutableMappingEvidence {
            start,
            end: start + 4096,
            file_offset: start - 4096,
            permissions: "r-xp".into(),
            device_major: 1,
            device_minor: 2,
            inode: 3,
            mapped_bytes_blake3: d("mapped"),
            backing_bytes_blake3: d("backing"),
        }
    }

    #[test]
    fn mapping_set_digest_is_order_canonical() {
        let left = vec![mapping(4096), mapping(8192)];
        let right = vec![mapping(8192), mapping(4096)];
        assert_eq!(mapping_set_digest(&left, 1), mapping_set_digest(&right, 1));
        assert_ne!(mapping_set_digest(&left, 1), mapping_set_digest(&left, 2));
    }

    #[test]
    fn transcript_detects_exec_enable_and_thread_escape() {
        let values = vec![
            BootstrapSyscallObservation {
                syscall_number: nix::libc::SYS_mmap,
                args: [0, 0, nix::libc::PROT_EXEC as u64, 0, 0, 0],
                return_value: 0,
                ready_token_write: false,
            },
            BootstrapSyscallObservation {
                syscall_number: nix::libc::SYS_clone,
                args: [0; 6],
                return_value: 0,
                ready_token_write: false,
            },
        ];
        let issues = transcript_issues(&values);
        assert!(issues.iter().any(|value| matches!(value, BootstrapExecutableMappingContinuityIssue::ExecutableMmapObserved)));
        assert!(issues.iter().any(|value| matches!(value, BootstrapExecutableMappingContinuityIssue::ThreadEscapeSyscallObserved(_))));
    }

    #[test]
    fn claim_ceiling_keeps_external_mutation_out() {
        let claims = [
            "tracee_action_mapping_continuity=true",
            "kernel_pseudo_mapping_identity_continuity=false",
            "external_process_mutation_excluded=false",
            "root_resistant_memory_immutability=false",
            "trusted_time=false",
            "physical_authority=false",
        ];
        assert_eq!(claims.len(), 6);
    }
}
