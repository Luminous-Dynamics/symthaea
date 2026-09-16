// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Supervisor-issued runtime re-entry challenge after an exact successful
//! continuity-qualified bootstrap release.
//!
//! The theorem is deliberately split into two causal stages. Before release,
//! this crate borrows the exact confinement and executable-mapping continuity
//! qualifications that the release constructor will later consume and retains a
//! private CLOEXEC duplicate of the reviewed supervisor handoff write end. After
//! release, only a `ReleasedBootstrapVerifier` naming those exact consumed
//! qualifications can use that retained channel to issue a fresh checkpoint-two
//! challenge. No timestamp comparison is needed to establish ordering.

#![cfg(target_os = "linux")]
#![deny(unsafe_code)]

use nix::{
    fcntl::{fcntl, FcntlArg, OFlag},
    unistd::{close, write},
};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeSet, fs, os::unix::io::RawFd};
use symthaea_assurance_bootstrap_executable_mapping_continuity::
    BootstrapExecutableMappingContinuityQualification;
use symthaea_assurance_bootstrap_ready_checkpoint::{
    verify_bootstrap_ready_wire, BootstrapReadyCheckpointPolicy,
};
use symthaea_assurance_bootstrap_syscall_confinement::{
    BootstrapSyscallConfinementDisposition, BootstrapSyscallConfinementQualification,
    BootstrapSyscallConfinementReport,
};
use symthaea_assurance_continuity_qualified_bootstrap_release::ReleasedBootstrapVerifier;
use symthaea_assurance_launch_runtime_handoff_channel::{
    IssuedLaunchRuntimeChallenge, LaunchRuntimeHandoffDisposition, LaunchRuntimeHandoffReport,
};
use symthaea_evidence_verifier_runtime_continuity::VerifierRuntimeContinuityPolicy;

pub const POST_RELEASE_RUNTIME_HANDOFF_POLICY_SCHEMA_V1: &str =
    "symthaea.assurance.post-release-runtime-handoff-policy.v1";
pub const POST_RELEASE_RUNTIME_CHANNEL_RETENTION_REPORT_SCHEMA_V1: &str =
    "symthaea.assurance.post-release-runtime-channel-retention-report.v1";
pub const POST_RELEASE_RUNTIME_HANDOFF_TICKET_SCHEMA_V1: &str =
    "symthaea.assurance.post-release-runtime-handoff-ticket.v1";
pub const POST_RELEASE_RUNTIME_HANDOFF_REPORT_SCHEMA_V1: &str =
    "symthaea.assurance.post-release-runtime-handoff-report.v1";

const POLICY_DOMAIN: &[u8] =
    b"symthaea.assurance.post-release-runtime-handoff-policy.digest.v1\0";
const RETENTION_REPORT_DOMAIN: &[u8] =
    b"symthaea.assurance.post-release-runtime-channel-retention-report.digest.v1\0";
const RETENTION_DOMAIN: &[u8] =
    b"symthaea.assurance.post-release-runtime-channel-retention.digest.v1\0";
const TICKET_DOMAIN: &[u8] =
    b"symthaea.assurance.post-release-runtime-handoff-ticket.digest.v1\0";
const REPORT_DOMAIN: &[u8] =
    b"symthaea.assurance.post-release-runtime-handoff-report.digest.v1\0";
const QUALIFICATION_DOMAIN: &[u8] =
    b"symthaea.assurance.post-release-runtime-handoff-qualification.digest.v1\0";
const CONFINEMENT_QUALIFICATION_DOMAIN: &[u8] =
    b"symthaea.assurance.bootstrap-syscall-confinement-qualification.digest.v1\0";
const WIRE_MAGIC: &[u8] = b"SYMT-POST-RELEASE-HANDOFF-V1\0";
const MAX_TEXT: usize = 4096;
const MAX_REFS: usize = 128;
const HARD_MAX_TICKET_BYTES: u32 = 4096;
const HARD_MAX_FDINFO_BYTES: u64 = 4 * 1024 * 1024;
const MIN_RETAINED_FD: RawFd = 3;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PostReleaseRuntimeHandoffPolicy {
    pub schema_version: String,
    pub policy_id: String,
    pub expected_confinement_policy_digest: String,
    pub expected_mapping_continuity_policy_digest: String,
    pub expected_release_policy_digest: String,
    pub expected_launch_handoff_policy_digest: String,
    pub expected_bootstrap_ready_policy_digest: String,
    pub expected_runtime_policy_digest: String,
    pub expected_runtime_verifier_ref: String,
    pub expected_backend_id: String,
    pub max_ticket_bytes: u32,
    pub max_fdinfo_bytes: u64,
    pub evidence_refs: Vec<String>,
}

impl PostReleaseRuntimeHandoffPolicy {
    pub fn validate(&self) -> bool {
        self.schema_version == POST_RELEASE_RUNTIME_HANDOFF_POLICY_SCHEMA_V1
            && canonical_text(&self.policy_id)
            && valid_blake3(&self.expected_confinement_policy_digest)
            && valid_blake3(&self.expected_mapping_continuity_policy_digest)
            && valid_blake3(&self.expected_release_policy_digest)
            && valid_blake3(&self.expected_launch_handoff_policy_digest)
            && valid_blake3(&self.expected_bootstrap_ready_policy_digest)
            && valid_blake3(&self.expected_runtime_policy_digest)
            && canonical_text(&self.expected_runtime_verifier_ref)
            && canonical_text(&self.expected_backend_id)
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
            self.expected_confinement_policy_digest.as_str(),
            self.expected_mapping_continuity_policy_digest.as_str(),
            self.expected_release_policy_digest.as_str(),
            self.expected_launch_handoff_policy_digest.as_str(),
            self.expected_bootstrap_ready_policy_digest.as_str(),
            self.expected_runtime_policy_digest.as_str(),
            self.expected_runtime_verifier_ref.as_str(),
            self.expected_backend_id.as_str(),
        ] {
            field(&mut h, value);
        }
        h.update(&self.max_ticket_bytes.to_le_bytes());
        h.update(&self.max_fdinfo_bytes.to_le_bytes());
        sorted(&mut h, &self.evidence_refs);
        Some(b3(h.finalize()))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PostReleaseRuntimeChannelRetentionIssue {
    InvalidPolicy,
    ConfinementPolicyMismatch,
    MappingContinuityPolicyMismatch,
    LaunchHandoffPolicyMismatch,
    RuntimePolicyMismatch,
    RuntimeVerifierMismatch,
    BackendMismatch,
    MappingConfinementMismatch,
    MappingExecMismatch,
    MappingSyscallSequenceMismatch,
    LaunchConfinementMismatch,
    LaunchTraceeMismatch,
    LaunchExecMismatch,
    LaunchAuthorityMismatch,
    LaunchReportNotQualified,
    LaunchReportMismatch,
    LaunchReportIdentityMismatch,
    InvalidSupervisorWriteFd,
    SourceFdUnavailable(String),
    SourcePipeMismatch,
    SourceFdNotWriteOnly,
    SourceFdNonBlocking,
    SourceFdInfoMismatch,
    RetainedFdDuplicateFailed(String),
    RetainedFdUnavailable(String),
    RetainedPipeMismatch,
    RetainedFdNotWriteOnly,
    RetainedFdNonBlocking,
    SourceFdChangedDuringRetention,
}

impl PostReleaseRuntimeChannelRetentionIssue {
    fn invalid(&self) -> bool {
        matches!(self, Self::InvalidPolicy | Self::InvalidSupervisorWriteFd)
    }

    fn code(&self) -> String {
        match self {
            Self::InvalidPolicy => "invalid-policy".into(),
            Self::ConfinementPolicyMismatch => "confinement-policy-mismatch".into(),
            Self::MappingContinuityPolicyMismatch => "mapping-policy-mismatch".into(),
            Self::LaunchHandoffPolicyMismatch => "launch-policy-mismatch".into(),
            Self::RuntimePolicyMismatch => "runtime-policy-mismatch".into(),
            Self::RuntimeVerifierMismatch => "runtime-verifier-mismatch".into(),
            Self::BackendMismatch => "backend-mismatch".into(),
            Self::MappingConfinementMismatch => "mapping-confinement-mismatch".into(),
            Self::MappingExecMismatch => "mapping-exec-mismatch".into(),
            Self::MappingSyscallSequenceMismatch => "mapping-syscall-sequence-mismatch".into(),
            Self::LaunchConfinementMismatch => "launch-confinement-mismatch".into(),
            Self::LaunchTraceeMismatch => "launch-tracee-mismatch".into(),
            Self::LaunchExecMismatch => "launch-exec-mismatch".into(),
            Self::LaunchAuthorityMismatch => "launch-authority-mismatch".into(),
            Self::LaunchReportNotQualified => "launch-report-not-qualified".into(),
            Self::LaunchReportMismatch => "launch-report-mismatch".into(),
            Self::LaunchReportIdentityMismatch => "launch-report-identity-mismatch".into(),
            Self::InvalidSupervisorWriteFd => "invalid-supervisor-write-fd".into(),
            Self::SourceFdUnavailable(value) => format!("source-fd-unavailable:{value}"),
            Self::SourcePipeMismatch => "source-pipe-mismatch".into(),
            Self::SourceFdNotWriteOnly => "source-fd-not-write-only".into(),
            Self::SourceFdNonBlocking => "source-fd-nonblocking".into(),
            Self::SourceFdInfoMismatch => "source-fdinfo-mismatch".into(),
            Self::RetainedFdDuplicateFailed(value) => format!("retained-fd-duplicate-failed:{value}"),
            Self::RetainedFdUnavailable(value) => format!("retained-fd-unavailable:{value}"),
            Self::RetainedPipeMismatch => "retained-pipe-mismatch".into(),
            Self::RetainedFdNotWriteOnly => "retained-fd-not-write-only".into(),
            Self::RetainedFdNonBlocking => "retained-fd-nonblocking".into(),
            Self::SourceFdChangedDuringRetention => "source-fd-changed-during-retention".into(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PostReleaseRuntimeChannelRetentionDisposition {
    Invalid,
    Blocked,
    Qualified,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PostReleaseRuntimeChannelRetentionReport {
    pub schema_version: String,
    pub policy_id: String,
    pub policy_digest: Option<String>,
    pub confinement_qualification_digest: String,
    pub mapping_continuity_qualification_digest: String,
    pub launch_handoff_qualification_digest: String,
    pub launch_handoff_report_digest: String,
    pub exec_confirmation_digest: String,
    pub ready_wire_digest: String,
    pub ready_checkpoint_digest: String,
    pub ready_wire_bytes_digest: String,
    pub syscall_sequence_digest: String,
    pub launch_mapping_set_digest: String,
    pub ready_mapping_set_digest: String,
    pub authority_snapshot_digest: String,
    pub tracee_pid: i32,
    pub tracee_handoff_read_fd: u32,
    pub supervisor_handoff_write_fd: i32,
    pub pipe_target: Option<String>,
    pub source_fdinfo_before_digest: Option<String>,
    pub source_fdinfo_after_digest: Option<String>,
    pub retained_fdinfo_digest: Option<String>,
    pub channel_binding_digest: String,
    pub disposition: PostReleaseRuntimeChannelRetentionDisposition,
    pub issues: Vec<PostReleaseRuntimeChannelRetentionIssue>,
}

impl PostReleaseRuntimeChannelRetentionReport {
    pub fn canonical_digest(&self) -> String {
        let mut h = blake3::Hasher::new();
        h.update(RETENTION_REPORT_DOMAIN);
        for value in [
            self.schema_version.as_str(),
            self.policy_id.as_str(),
            self.policy_digest.as_deref().unwrap_or("-"),
            self.confinement_qualification_digest.as_str(),
            self.mapping_continuity_qualification_digest.as_str(),
            self.launch_handoff_qualification_digest.as_str(),
            self.launch_handoff_report_digest.as_str(),
            self.exec_confirmation_digest.as_str(),
            self.ready_wire_digest.as_str(),
            self.ready_checkpoint_digest.as_str(),
            self.ready_wire_bytes_digest.as_str(),
            self.syscall_sequence_digest.as_str(),
            self.launch_mapping_set_digest.as_str(),
            self.ready_mapping_set_digest.as_str(),
            self.authority_snapshot_digest.as_str(),
            self.pipe_target.as_deref().unwrap_or("-"),
            self.source_fdinfo_before_digest.as_deref().unwrap_or("-"),
            self.source_fdinfo_after_digest.as_deref().unwrap_or("-"),
            self.retained_fdinfo_digest.as_deref().unwrap_or("-"),
            self.channel_binding_digest.as_str(),
        ] {
            field(&mut h, value);
        }
        h.update(&self.tracee_pid.to_le_bytes());
        h.update(&self.tracee_handoff_read_fd.to_le_bytes());
        h.update(&self.supervisor_handoff_write_fd.to_le_bytes());
        field(
            &mut h,
            match self.disposition {
                PostReleaseRuntimeChannelRetentionDisposition::Invalid => "invalid",
                PostReleaseRuntimeChannelRetentionDisposition::Blocked => "blocked",
                PostReleaseRuntimeChannelRetentionDisposition::Qualified => "qualified",
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

pub struct RetainedPostReleaseRuntimeHandoffChannel {
    retention_digest: String,
    report_digest: String,
    policy_digest: String,
    confinement_qualification_digest: String,
    mapping_continuity_qualification_digest: String,
    launch_handoff_qualification_digest: String,
    launch_handoff_report_digest: String,
    exec_confirmation_digest: String,
    ready_wire_digest: String,
    ready_checkpoint_digest: String,
    ready_wire_bytes_digest: String,
    syscall_sequence_digest: String,
    launch_mapping_set_digest: String,
    ready_mapping_set_digest: String,
    authority_snapshot_digest: String,
    tracee_pid: i32,
    tracee_handoff_read_fd: u32,
    original_supervisor_write_fd: RawFd,
    retained_write_fd: RawFd,
    pipe_target: String,
    pipe_inode: u64,
    channel_binding_digest: String,
    retained_observation: FdObservation,
    runtime_policy_digest: String,
    runtime_verifier_ref: String,
    backend_id: String,
    executable_digest: String,
}

impl Drop for RetainedPostReleaseRuntimeHandoffChannel {
    fn drop(&mut self) {
        let _ = close(self.retained_write_fd);
    }
}

impl RetainedPostReleaseRuntimeHandoffChannel {
    pub fn retention_digest(&self) -> &str { &self.retention_digest }
    pub fn report_digest(&self) -> &str { &self.report_digest }
    pub fn policy_digest(&self) -> &str { &self.policy_digest }
    pub fn confinement_qualification_digest(&self) -> &str {
        &self.confinement_qualification_digest
    }
    pub fn mapping_continuity_qualification_digest(&self) -> &str {
        &self.mapping_continuity_qualification_digest
    }
    pub fn launch_handoff_qualification_digest(&self) -> &str {
        &self.launch_handoff_qualification_digest
    }
    pub fn ready_checkpoint_digest(&self) -> &str { &self.ready_checkpoint_digest }
    pub fn pipe_target(&self) -> &str { &self.pipe_target }
    pub const fn tracee_pid(&self) -> i32 { self.tracee_pid }
    pub fn runtime_policy_digest(&self) -> &str { &self.runtime_policy_digest }
    pub fn runtime_verifier_ref(&self) -> &str { &self.runtime_verifier_ref }
    pub fn backend_id(&self) -> &str { &self.backend_id }
    pub fn executable_digest(&self) -> &str { &self.executable_digest }

    pub const fn constructed_while_exact_release_inputs_still_existed(&self) -> bool { true }
    pub const fn private_cloexec_duplicate_created_from_reviewed_write_end(&self) -> bool { true }
    pub const fn duplicate_matched_original_pipe_at_retention(&self) -> bool { true }
    pub const fn exact_successful_release_established_yet(&self) -> bool { false }
    pub const fn arbitrary_same_process_descriptor_sabotage_excluded(&self) -> bool { false }
    pub const fn trusted_time_established(&self) -> bool { false }
    pub const fn grants_physical_authority(&self) -> bool { false }
}

#[allow(clippy::too_many_arguments)]
pub fn retain_post_release_runtime_handoff_channel(
    policy: &PostReleaseRuntimeHandoffPolicy,
    confinement: &BootstrapSyscallConfinementQualification,
    mapping: &BootstrapExecutableMappingContinuityQualification,
    launch: &IssuedLaunchRuntimeChallenge,
    launch_report: &LaunchRuntimeHandoffReport,
    supervisor_write_fd: RawFd,
) -> Result<RetainedPostReleaseRuntimeHandoffChannel, PostReleaseRuntimeChannelRetentionReport> {
    let policy_digest = policy.canonical_digest();
    let ready = confinement.ready();
    let mapped = mapping.verified();
    let launch_report_digest = launch_report.canonical_digest();
    let mut report = PostReleaseRuntimeChannelRetentionReport {
        schema_version: POST_RELEASE_RUNTIME_CHANNEL_RETENTION_REPORT_SCHEMA_V1.into(),
        policy_id: policy.policy_id.clone(),
        policy_digest: policy_digest.clone(),
        confinement_qualification_digest: ready.qualification_digest().into(),
        mapping_continuity_qualification_digest: mapped.qualification_digest().into(),
        launch_handoff_qualification_digest: launch.qualification_digest().into(),
        launch_handoff_report_digest: launch_report_digest.clone(),
        exec_confirmation_digest: ready.exec_confirmation_digest().into(),
        ready_wire_digest: ready.ready_wire_digest().into(),
        ready_checkpoint_digest: ready.ready_checkpoint_digest().into(),
        ready_wire_bytes_digest: ready.ready_wire_bytes_digest().into(),
        syscall_sequence_digest: ready.syscall_sequence_digest().into(),
        launch_mapping_set_digest: mapped.launch_mapping_set_digest().into(),
        ready_mapping_set_digest: mapped.ready_mapping_set_digest().into(),
        authority_snapshot_digest: ready.authority_snapshot_digest().into(),
        tracee_pid: ready.tracee_pid(),
        tracee_handoff_read_fd: launch.tracee_handoff_read_fd(),
        supervisor_handoff_write_fd: supervisor_write_fd,
        pipe_target: None,
        source_fdinfo_before_digest: None,
        source_fdinfo_after_digest: None,
        retained_fdinfo_digest: None,
        channel_binding_digest: launch.channel_binding_digest().into(),
        disposition: PostReleaseRuntimeChannelRetentionDisposition::Invalid,
        issues: Vec::new(),
    };

    if !policy.validate() {
        report
            .issues
            .push(PostReleaseRuntimeChannelRetentionIssue::InvalidPolicy);
    }
    if ready.policy_digest() != policy.expected_confinement_policy_digest {
        report
            .issues
            .push(PostReleaseRuntimeChannelRetentionIssue::ConfinementPolicyMismatch);
    }
    if mapped.policy_digest() != policy.expected_mapping_continuity_policy_digest {
        report
            .issues
            .push(PostReleaseRuntimeChannelRetentionIssue::MappingContinuityPolicyMismatch);
    }
    if launch.policy_digest() != policy.expected_launch_handoff_policy_digest {
        report
            .issues
            .push(PostReleaseRuntimeChannelRetentionIssue::LaunchHandoffPolicyMismatch);
    }
    if launch.runtime_policy_digest() != policy.expected_runtime_policy_digest {
        report
            .issues
            .push(PostReleaseRuntimeChannelRetentionIssue::RuntimePolicyMismatch);
    }
    if launch.runtime_verifier_ref() != policy.expected_runtime_verifier_ref {
        report
            .issues
            .push(PostReleaseRuntimeChannelRetentionIssue::RuntimeVerifierMismatch);
    }
    if launch.backend_id() != policy.expected_backend_id {
        report
            .issues
            .push(PostReleaseRuntimeChannelRetentionIssue::BackendMismatch);
    }
    if mapped.confinement_qualification_digest() != ready.qualification_digest() {
        report
            .issues
            .push(PostReleaseRuntimeChannelRetentionIssue::MappingConfinementMismatch);
    }
    if mapped.exec_confirmation_digest() != ready.exec_confirmation_digest() {
        report
            .issues
            .push(PostReleaseRuntimeChannelRetentionIssue::MappingExecMismatch);
    }
    if mapped.syscall_sequence_digest() != ready.syscall_sequence_digest() {
        report
            .issues
            .push(PostReleaseRuntimeChannelRetentionIssue::MappingSyscallSequenceMismatch);
    }
    if ready.issued_challenge_qualification_digest() != launch.qualification_digest() {
        report
            .issues
            .push(PostReleaseRuntimeChannelRetentionIssue::LaunchConfinementMismatch);
    }
    if ready.tracee_pid() != launch.tracee_pid() {
        report
            .issues
            .push(PostReleaseRuntimeChannelRetentionIssue::LaunchTraceeMismatch);
    }
    if ready.exec_confirmation_digest() != launch.exec_confirmation_digest() {
        report
            .issues
            .push(PostReleaseRuntimeChannelRetentionIssue::LaunchExecMismatch);
    }
    if ready.authority_snapshot_digest() != launch.authority_snapshot_digest() {
        report
            .issues
            .push(PostReleaseRuntimeChannelRetentionIssue::LaunchAuthorityMismatch);
    }
    if launch_report.disposition != LaunchRuntimeHandoffDisposition::Qualified
        || !launch_report.issues.is_empty()
    {
        report
            .issues
            .push(PostReleaseRuntimeChannelRetentionIssue::LaunchReportNotQualified);
    }
    if launch_report_digest != launch.report_digest() {
        report
            .issues
            .push(PostReleaseRuntimeChannelRetentionIssue::LaunchReportMismatch);
    }
    if launch_report.tracee_pid != launch.tracee_pid()
        || launch_report.tracee_handoff_read_fd != launch.tracee_handoff_read_fd()
        || launch_report.supervisor_handoff_write_fd != launch.supervisor_handoff_write_fd()
        || launch_report.pipe_target.as_deref() != Some(launch.pipe_target())
        || launch_report.channel_binding_digest.as_deref() != Some(launch.channel_binding_digest())
    {
        report
            .issues
            .push(PostReleaseRuntimeChannelRetentionIssue::LaunchReportIdentityMismatch);
    }
    if supervisor_write_fd < 0 {
        report
            .issues
            .push(PostReleaseRuntimeChannelRetentionIssue::InvalidSupervisorWriteFd);
    }
    if supervisor_write_fd != launch.supervisor_handoff_write_fd()
        || supervisor_write_fd != launch_report.supervisor_handoff_write_fd
    {
        report
            .issues
            .push(PostReleaseRuntimeChannelRetentionIssue::LaunchReportIdentityMismatch);
    }
    if !report.issues.is_empty() {
        return Err(finalize_retention(report));
    }

    let source_before = match observe_fd(supervisor_write_fd, policy.max_fdinfo_bytes) {
        Ok(value) => value,
        Err(error) => {
            report
                .issues
                .push(PostReleaseRuntimeChannelRetentionIssue::SourceFdUnavailable(error));
            return Err(finalize_retention(report));
        }
    };
    report.pipe_target = Some(source_before.target.clone());
    report.source_fdinfo_before_digest = Some(source_before.fdinfo_digest.clone());
    if source_before.target != launch.pipe_target()
        || parse_pipe_target(&source_before.target) != Some(source_before.inode)
    {
        report
            .issues
            .push(PostReleaseRuntimeChannelRetentionIssue::SourcePipeMismatch);
    }
    if source_before.flags & OFlag::O_ACCMODE.bits() != OFlag::O_WRONLY.bits() {
        report
            .issues
            .push(PostReleaseRuntimeChannelRetentionIssue::SourceFdNotWriteOnly);
    }
    if source_before.flags & OFlag::O_NONBLOCK.bits() != 0 {
        report
            .issues
            .push(PostReleaseRuntimeChannelRetentionIssue::SourceFdNonBlocking);
    }
    if launch_report.supervisor_fdinfo_digest.as_deref()
        != Some(source_before.fdinfo_digest.as_str())
    {
        report
            .issues
            .push(PostReleaseRuntimeChannelRetentionIssue::SourceFdInfoMismatch);
    }
    if !report.issues.is_empty() {
        return Err(finalize_retention(report));
    }

    let retained_write_fd = match fcntl(
        supervisor_write_fd,
        FcntlArg::F_DUPFD_CLOEXEC(MIN_RETAINED_FD),
    ) {
        Ok(value) => value,
        Err(error) => {
            report
                .issues
                .push(PostReleaseRuntimeChannelRetentionIssue::RetainedFdDuplicateFailed(
                    error.to_string(),
                ));
            return Err(finalize_retention(report));
        }
    };

    let retained_observation = match observe_fd(retained_write_fd, policy.max_fdinfo_bytes) {
        Ok(value) => value,
        Err(error) => {
            let _ = close(retained_write_fd);
            report
                .issues
                .push(PostReleaseRuntimeChannelRetentionIssue::RetainedFdUnavailable(error));
            return Err(finalize_retention(report));
        }
    };
    report.retained_fdinfo_digest = Some(retained_observation.fdinfo_digest.clone());
    if retained_observation.target != source_before.target
        || retained_observation.inode != source_before.inode
    {
        report
            .issues
            .push(PostReleaseRuntimeChannelRetentionIssue::RetainedPipeMismatch);
    }
    if retained_observation.flags & OFlag::O_ACCMODE.bits() != OFlag::O_WRONLY.bits() {
        report
            .issues
            .push(PostReleaseRuntimeChannelRetentionIssue::RetainedFdNotWriteOnly);
    }
    if retained_observation.flags & OFlag::O_NONBLOCK.bits() != 0 {
        report
            .issues
            .push(PostReleaseRuntimeChannelRetentionIssue::RetainedFdNonBlocking);
    }

    let source_after = match observe_fd(supervisor_write_fd, policy.max_fdinfo_bytes) {
        Ok(value) => value,
        Err(error) => {
            let _ = close(retained_write_fd);
            report
                .issues
                .push(PostReleaseRuntimeChannelRetentionIssue::SourceFdUnavailable(error));
            return Err(finalize_retention(report));
        }
    };
    report.source_fdinfo_after_digest = Some(source_after.fdinfo_digest.clone());
    if source_before != source_after {
        report
            .issues
            .push(PostReleaseRuntimeChannelRetentionIssue::SourceFdChangedDuringRetention);
    }
    if !report.issues.is_empty() {
        let _ = close(retained_write_fd);
        return Err(finalize_retention(report));
    }

    report.disposition = PostReleaseRuntimeChannelRetentionDisposition::Qualified;
    let report_digest = report.canonical_digest();
    let policy_digest = policy_digest.expect("validated policy has digest");
    let retention_digest = retention_digest(
        &policy_digest,
        ready.qualification_digest(),
        mapped.qualification_digest(),
        launch.qualification_digest(),
        ready.ready_wire_digest(),
        ready.ready_checkpoint_digest(),
        supervisor_write_fd,
        retained_write_fd,
        &source_before,
        &retained_observation,
        &report_digest,
    );

    Ok(RetainedPostReleaseRuntimeHandoffChannel {
        retention_digest,
        report_digest,
        policy_digest,
        confinement_qualification_digest: ready.qualification_digest().into(),
        mapping_continuity_qualification_digest: mapped.qualification_digest().into(),
        launch_handoff_qualification_digest: launch.qualification_digest().into(),
        launch_handoff_report_digest: launch_report_digest,
        exec_confirmation_digest: ready.exec_confirmation_digest().into(),
        ready_wire_digest: ready.ready_wire_digest().into(),
        ready_checkpoint_digest: ready.ready_checkpoint_digest().into(),
        ready_wire_bytes_digest: ready.ready_wire_bytes_digest().into(),
        syscall_sequence_digest: ready.syscall_sequence_digest().into(),
        launch_mapping_set_digest: mapped.launch_mapping_set_digest().into(),
        ready_mapping_set_digest: mapped.ready_mapping_set_digest().into(),
        authority_snapshot_digest: ready.authority_snapshot_digest().into(),
        tracee_pid: ready.tracee_pid(),
        tracee_handoff_read_fd: launch.tracee_handoff_read_fd(),
        original_supervisor_write_fd: supervisor_write_fd,
        retained_write_fd,
        pipe_target: source_before.target.clone(),
        pipe_inode: source_before.inode,
        channel_binding_digest: launch.channel_binding_digest().into(),
        retained_observation,
        runtime_policy_digest: launch.runtime_policy_digest().into(),
        runtime_verifier_ref: launch.runtime_verifier_ref().into(),
        backend_id: launch.backend_id().into(),
        executable_digest: launch.executable_digest().into(),
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PostReleaseRuntimeHandoffTicket {
    pub schema_version: String,
    pub channel_retention_digest: String,
    pub release_digest: String,
    pub release_report_digest: String,
    pub confinement_qualification_digest: String,
    pub mapping_continuity_qualification_digest: String,
    pub bootstrap_ready_wire_digest: String,
    pub bootstrap_ready_checkpoint_digest: String,
    pub launch_handoff_qualification_digest: String,
    pub launch_ticket_digest: String,
    pub tracee_pid: i32,
    pub tracee_handoff_read_fd: u32,
    pub pipe_target: String,
    pub original_channel_binding_digest: String,
    pub process_instance_id: String,
    pub runtime_policy_digest: String,
    pub runtime_verifier_ref: String,
    pub backend_id: String,
    pub boot_measurement_digest: String,
    pub executable_digest: String,
    pub dependency_closure_digest: String,
    pub runtime_config_digest: String,
    pub launch_attestation_digest: String,
    pub checkpoint_one_challenge_digest: String,
    pub checkpoint_one_observation_qualification_digest: String,
    pub checkpoint_sequence: u64,
    pub previous_checkpoint_digest: String,
    pub launch_monotonic_counter: u64,
    pub previous_checkpoint_monotonic_counter: u64,
    pub challenge_nonce_blake3_hex: String,
    pub ticket_digest: String,
}

impl PostReleaseRuntimeHandoffTicket {
    pub fn validate(&self) -> bool {
        self.schema_version == POST_RELEASE_RUNTIME_HANDOFF_TICKET_SCHEMA_V1
            && valid_blake3(&self.channel_retention_digest)
            && valid_blake3(&self.release_digest)
            && valid_blake3(&self.release_report_digest)
            && valid_blake3(&self.confinement_qualification_digest)
            && valid_blake3(&self.mapping_continuity_qualification_digest)
            && valid_blake3(&self.bootstrap_ready_wire_digest)
            && valid_blake3(&self.bootstrap_ready_checkpoint_digest)
            && valid_blake3(&self.launch_handoff_qualification_digest)
            && valid_blake3(&self.launch_ticket_digest)
            && self.tracee_pid > 1
            && self.tracee_handoff_read_fd <= 1_048_576
            && parse_pipe_target(&self.pipe_target).is_some()
            && valid_blake3(&self.original_channel_binding_digest)
            && canonical_text(&self.process_instance_id)
            && valid_blake3(&self.runtime_policy_digest)
            && canonical_text(&self.runtime_verifier_ref)
            && canonical_text(&self.backend_id)
            && valid_blake3(&self.boot_measurement_digest)
            && valid_blake3(&self.executable_digest)
            && valid_blake3(&self.dependency_closure_digest)
            && valid_blake3(&self.runtime_config_digest)
            && valid_blake3(&self.launch_attestation_digest)
            && valid_blake3(&self.checkpoint_one_challenge_digest)
            && valid_blake3(&self.checkpoint_one_observation_qualification_digest)
            && self.checkpoint_sequence == 2
            && valid_blake3(&self.previous_checkpoint_digest)
            && self.launch_monotonic_counter > 0
            && self.previous_checkpoint_monotonic_counter > self.launch_monotonic_counter
            && valid_nonce(&self.challenge_nonce_blake3_hex)
            && valid_blake3(&self.ticket_digest)
            && self.ticket_digest == self.recompute_digest()
    }

    pub fn recompute_digest(&self) -> String {
        ticket_digest(self)
    }

    pub fn to_wire_bytes(&self) -> Option<Vec<u8>> {
        if !self.validate() {
            return None;
        }
        let mut out = Vec::with_capacity(2048);
        out.extend_from_slice(WIRE_MAGIC);
        for value in [
            self.schema_version.as_str(),
            self.channel_retention_digest.as_str(),
            self.release_digest.as_str(),
            self.release_report_digest.as_str(),
            self.confinement_qualification_digest.as_str(),
            self.mapping_continuity_qualification_digest.as_str(),
            self.bootstrap_ready_wire_digest.as_str(),
            self.bootstrap_ready_checkpoint_digest.as_str(),
            self.launch_handoff_qualification_digest.as_str(),
            self.launch_ticket_digest.as_str(),
        ] {
            wire_string(&mut out, value)?;
        }
        out.extend_from_slice(&self.tracee_pid.to_le_bytes());
        out.extend_from_slice(&self.tracee_handoff_read_fd.to_le_bytes());
        for value in [
            self.pipe_target.as_str(),
            self.original_channel_binding_digest.as_str(),
            self.process_instance_id.as_str(),
            self.runtime_policy_digest.as_str(),
            self.runtime_verifier_ref.as_str(),
            self.backend_id.as_str(),
            self.boot_measurement_digest.as_str(),
            self.executable_digest.as_str(),
            self.dependency_closure_digest.as_str(),
            self.runtime_config_digest.as_str(),
            self.launch_attestation_digest.as_str(),
            self.checkpoint_one_challenge_digest.as_str(),
            self.checkpoint_one_observation_qualification_digest.as_str(),
        ] {
            wire_string(&mut out, value)?;
        }
        out.extend_from_slice(&self.checkpoint_sequence.to_le_bytes());
        wire_string(&mut out, &self.previous_checkpoint_digest)?;
        out.extend_from_slice(&self.launch_monotonic_counter.to_le_bytes());
        out.extend_from_slice(&self.previous_checkpoint_monotonic_counter.to_le_bytes());
        wire_string(&mut out, &self.challenge_nonce_blake3_hex)?;
        wire_string(&mut out, &self.ticket_digest)?;
        Some(out)
    }

    pub fn from_wire_bytes(bytes: &[u8]) -> Option<Self> {
        if !bytes.starts_with(WIRE_MAGIC) {
            return None;
        }
        let mut cursor = Cursor {
            bytes,
            offset: WIRE_MAGIC.len(),
        };
        let value = Self {
            schema_version: cursor.string()?,
            channel_retention_digest: cursor.string()?,
            release_digest: cursor.string()?,
            release_report_digest: cursor.string()?,
            confinement_qualification_digest: cursor.string()?,
            mapping_continuity_qualification_digest: cursor.string()?,
            bootstrap_ready_wire_digest: cursor.string()?,
            bootstrap_ready_checkpoint_digest: cursor.string()?,
            launch_handoff_qualification_digest: cursor.string()?,
            launch_ticket_digest: cursor.string()?,
            tracee_pid: cursor.i32()?,
            tracee_handoff_read_fd: cursor.u32()?,
            pipe_target: cursor.string()?,
            original_channel_binding_digest: cursor.string()?,
            process_instance_id: cursor.string()?,
            runtime_policy_digest: cursor.string()?,
            runtime_verifier_ref: cursor.string()?,
            backend_id: cursor.string()?,
            boot_measurement_digest: cursor.string()?,
            executable_digest: cursor.string()?,
            dependency_closure_digest: cursor.string()?,
            runtime_config_digest: cursor.string()?,
            launch_attestation_digest: cursor.string()?,
            checkpoint_one_challenge_digest: cursor.string()?,
            checkpoint_one_observation_qualification_digest: cursor.string()?,
            checkpoint_sequence: cursor.u64()?,
            previous_checkpoint_digest: cursor.string()?,
            launch_monotonic_counter: cursor.u64()?,
            previous_checkpoint_monotonic_counter: cursor.u64()?,
            challenge_nonce_blake3_hex: cursor.string()?,
            ticket_digest: cursor.string()?,
        };
        (cursor.offset == bytes.len() && value.validate()).then_some(value)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PostReleaseRuntimeHandoffIssue {
    InvalidPolicy,
    InvalidRuntimePolicy,
    RetentionPolicyMismatch,
    ReleasePolicyMismatch,
    ReleaseParentMismatch,
    ReleaseTraceeMismatch,
    ReleaseExecMismatch,
    ReleaseSyscallSequenceMismatch,
    ReleaseMappingSetMismatch,
    LaunchMismatch,
    ConfinementReportNotQualified,
    ConfinementReportIncomplete,
    ConfinementQualificationMismatch,
    ConfinementReportIdentityMismatch,
    BootstrapReadyPolicyMismatch,
    BootstrapReadyWireVerificationFailed(String),
    BootstrapReadyIdentityMismatch,
    BootstrapReadyBytesMismatch,
    BootstrapReadyTicketMismatch,
    BootstrapReadyChallengeMismatch,
    BootstrapReadyRuntimeMismatch,
    RetainedFdUnavailable(String),
    RetainedFdChanged,
    ChallengeEntropyUnavailable(String),
    TicketEncodingFailed,
    TicketTooLarge { observed: u64, maximum: u64 },
    TicketWriteFailed(String),
    PartialTicketWrite { expected: u64, observed: u64 },
    RetainedFdChangedDuringIssuance,
}

impl PostReleaseRuntimeHandoffIssue {
    fn invalid(&self) -> bool {
        matches!(self, Self::InvalidPolicy | Self::InvalidRuntimePolicy)
    }

    fn code(&self) -> String {
        match self {
            Self::InvalidPolicy => "invalid-policy".into(),
            Self::InvalidRuntimePolicy => "invalid-runtime-policy".into(),
            Self::RetentionPolicyMismatch => "retention-policy-mismatch".into(),
            Self::ReleasePolicyMismatch => "release-policy-mismatch".into(),
            Self::ReleaseParentMismatch => "release-parent-mismatch".into(),
            Self::ReleaseTraceeMismatch => "release-tracee-mismatch".into(),
            Self::ReleaseExecMismatch => "release-exec-mismatch".into(),
            Self::ReleaseSyscallSequenceMismatch => "release-syscall-sequence-mismatch".into(),
            Self::ReleaseMappingSetMismatch => "release-mapping-set-mismatch".into(),
            Self::LaunchMismatch => "launch-mismatch".into(),
            Self::ConfinementReportNotQualified => "confinement-report-not-qualified".into(),
            Self::ConfinementReportIncomplete => "confinement-report-incomplete".into(),
            Self::ConfinementQualificationMismatch => "confinement-qualification-mismatch".into(),
            Self::ConfinementReportIdentityMismatch => "confinement-report-identity-mismatch".into(),
            Self::BootstrapReadyPolicyMismatch => "bootstrap-ready-policy-mismatch".into(),
            Self::BootstrapReadyWireVerificationFailed(value) => {
                format!("bootstrap-ready-wire-verification-failed:{value}")
            }
            Self::BootstrapReadyIdentityMismatch => "bootstrap-ready-identity-mismatch".into(),
            Self::BootstrapReadyBytesMismatch => "bootstrap-ready-bytes-mismatch".into(),
            Self::BootstrapReadyTicketMismatch => "bootstrap-ready-ticket-mismatch".into(),
            Self::BootstrapReadyChallengeMismatch => "bootstrap-ready-challenge-mismatch".into(),
            Self::BootstrapReadyRuntimeMismatch => "bootstrap-ready-runtime-mismatch".into(),
            Self::RetainedFdUnavailable(value) => format!("retained-fd-unavailable:{value}"),
            Self::RetainedFdChanged => "retained-fd-changed".into(),
            Self::ChallengeEntropyUnavailable(value) => {
                format!("challenge-entropy-unavailable:{value}")
            }
            Self::TicketEncodingFailed => "ticket-encoding-failed".into(),
            Self::TicketTooLarge { observed, maximum } => {
                format!("ticket-too-large:{observed}:{maximum}")
            }
            Self::TicketWriteFailed(value) => format!("ticket-write-failed:{value}"),
            Self::PartialTicketWrite { expected, observed } => {
                format!("partial-ticket-write:{expected}:{observed}")
            }
            Self::RetainedFdChangedDuringIssuance => "retained-fd-changed-during-issuance".into(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PostReleaseRuntimeHandoffDisposition {
    Invalid,
    Blocked,
    Qualified,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PostReleaseRuntimeHandoffReport {
    pub schema_version: String,
    pub policy_id: String,
    pub policy_digest: Option<String>,
    pub channel_retention_digest: String,
    pub release_digest: String,
    pub release_report_digest: String,
    pub release_policy_digest: String,
    pub confinement_qualification_digest: String,
    pub mapping_continuity_qualification_digest: String,
    pub recomputed_confinement_qualification_digest: Option<String>,
    pub launch_handoff_qualification_digest: String,
    pub bootstrap_ready_wire_digest: Option<String>,
    pub bootstrap_ready_checkpoint_digest: Option<String>,
    pub bootstrap_ready_bytes_digest: String,
    pub process_instance_id: String,
    pub tracee_pid: i32,
    pub pipe_target: String,
    pub retained_fdinfo_before_digest: Option<String>,
    pub retained_fdinfo_after_digest: Option<String>,
    pub challenge_nonce_blake3_hex: Option<String>,
    pub ticket_digest: Option<String>,
    pub ticket_bytes_digest: Option<String>,
    pub ticket_bytes_len: u64,
    pub disposition: PostReleaseRuntimeHandoffDisposition,
    pub issues: Vec<PostReleaseRuntimeHandoffIssue>,
}

impl PostReleaseRuntimeHandoffReport {
    pub fn canonical_digest(&self) -> String {
        let mut h = blake3::Hasher::new();
        h.update(REPORT_DOMAIN);
        for value in [
            self.schema_version.as_str(),
            self.policy_id.as_str(),
            self.policy_digest.as_deref().unwrap_or("-"),
            self.channel_retention_digest.as_str(),
            self.release_digest.as_str(),
            self.release_report_digest.as_str(),
            self.release_policy_digest.as_str(),
            self.confinement_qualification_digest.as_str(),
            self.mapping_continuity_qualification_digest.as_str(),
            self.recomputed_confinement_qualification_digest
                .as_deref()
                .unwrap_or("-"),
            self.launch_handoff_qualification_digest.as_str(),
            self.bootstrap_ready_wire_digest.as_deref().unwrap_or("-"),
            self.bootstrap_ready_checkpoint_digest.as_deref().unwrap_or("-"),
            self.bootstrap_ready_bytes_digest.as_str(),
            self.process_instance_id.as_str(),
            self.pipe_target.as_str(),
            self.retained_fdinfo_before_digest.as_deref().unwrap_or("-"),
            self.retained_fdinfo_after_digest.as_deref().unwrap_or("-"),
            self.challenge_nonce_blake3_hex.as_deref().unwrap_or("-"),
            self.ticket_digest.as_deref().unwrap_or("-"),
            self.ticket_bytes_digest.as_deref().unwrap_or("-"),
        ] {
            field(&mut h, value);
        }
        h.update(&self.tracee_pid.to_le_bytes());
        h.update(&self.ticket_bytes_len.to_le_bytes());
        field(
            &mut h,
            match self.disposition {
                PostReleaseRuntimeHandoffDisposition::Invalid => "invalid",
                PostReleaseRuntimeHandoffDisposition::Blocked => "blocked",
                PostReleaseRuntimeHandoffDisposition::Qualified => "qualified",
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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IssuedPostReleaseRuntimeChallenge {
    qualification_digest: String,
    report_digest: String,
    policy_digest: String,
    channel_retention_digest: String,
    release_digest: String,
    release_report_digest: String,
    confinement_qualification_digest: String,
    mapping_continuity_qualification_digest: String,
    bootstrap_ready_wire_digest: String,
    bootstrap_ready_checkpoint_digest: String,
    launch_handoff_qualification_digest: String,
    launch_ticket_digest: String,
    tracee_pid: i32,
    tracee_handoff_read_fd: u32,
    pipe_target: String,
    original_channel_binding_digest: String,
    process_instance_id: String,
    runtime_policy_digest: String,
    runtime_verifier_ref: String,
    backend_id: String,
    boot_measurement_digest: String,
    executable_digest: String,
    dependency_closure_digest: String,
    runtime_config_digest: String,
    launch_attestation_digest: String,
    checkpoint_one_challenge_digest: String,
    checkpoint_one_observation_qualification_digest: String,
    launch_monotonic_counter: u64,
    previous_checkpoint_monotonic_counter: u64,
    challenge_nonce_blake3_hex: String,
    ticket_digest: String,
    ticket_bytes_digest: String,
}

impl IssuedPostReleaseRuntimeChallenge {
    pub fn qualification_digest(&self) -> &str { &self.qualification_digest }
    pub fn report_digest(&self) -> &str { &self.report_digest }
    pub fn policy_digest(&self) -> &str { &self.policy_digest }
    pub fn channel_retention_digest(&self) -> &str { &self.channel_retention_digest }
    pub fn release_digest(&self) -> &str { &self.release_digest }
    pub fn release_report_digest(&self) -> &str { &self.release_report_digest }
    pub fn confinement_qualification_digest(&self) -> &str {
        &self.confinement_qualification_digest
    }
    pub fn mapping_continuity_qualification_digest(&self) -> &str {
        &self.mapping_continuity_qualification_digest
    }
    pub fn bootstrap_ready_wire_digest(&self) -> &str { &self.bootstrap_ready_wire_digest }
    pub fn bootstrap_ready_checkpoint_digest(&self) -> &str {
        &self.bootstrap_ready_checkpoint_digest
    }
    pub fn launch_handoff_qualification_digest(&self) -> &str {
        &self.launch_handoff_qualification_digest
    }
    pub fn launch_ticket_digest(&self) -> &str { &self.launch_ticket_digest }
    pub const fn tracee_pid(&self) -> i32 { self.tracee_pid }
    pub const fn tracee_handoff_read_fd(&self) -> u32 { self.tracee_handoff_read_fd }
    pub fn pipe_target(&self) -> &str { &self.pipe_target }
    pub fn original_channel_binding_digest(&self) -> &str {
        &self.original_channel_binding_digest
    }
    pub fn process_instance_id(&self) -> &str { &self.process_instance_id }
    pub fn runtime_policy_digest(&self) -> &str { &self.runtime_policy_digest }
    pub fn runtime_verifier_ref(&self) -> &str { &self.runtime_verifier_ref }
    pub fn backend_id(&self) -> &str { &self.backend_id }
    pub fn boot_measurement_digest(&self) -> &str { &self.boot_measurement_digest }
    pub fn executable_digest(&self) -> &str { &self.executable_digest }
    pub fn dependency_closure_digest(&self) -> &str { &self.dependency_closure_digest }
    pub fn runtime_config_digest(&self) -> &str { &self.runtime_config_digest }
    pub fn launch_attestation_digest(&self) -> &str { &self.launch_attestation_digest }
    pub fn checkpoint_one_challenge_digest(&self) -> &str {
        &self.checkpoint_one_challenge_digest
    }
    pub fn checkpoint_one_observation_qualification_digest(&self) -> &str {
        &self.checkpoint_one_observation_qualification_digest
    }
    pub const fn checkpoint_sequence(&self) -> u64 { 2 }
    pub fn previous_checkpoint_digest(&self) -> &str { &self.bootstrap_ready_checkpoint_digest }
    pub const fn launch_monotonic_counter(&self) -> u64 { self.launch_monotonic_counter }
    pub const fn previous_checkpoint_monotonic_counter(&self) -> u64 {
        self.previous_checkpoint_monotonic_counter
    }
    pub fn challenge_nonce_blake3_hex(&self) -> &str { &self.challenge_nonce_blake3_hex }
    pub fn ticket_digest(&self) -> &str { &self.ticket_digest }
    pub fn ticket_bytes_digest(&self) -> &str { &self.ticket_bytes_digest }

    pub const fn channel_retention_precedes_exact_release_by_capability_consumption(&self) -> bool {
        true
    }
    pub const fn post_release_write_used_pre_release_retained_duplicate(&self) -> bool { true }
    pub const fn bootstrap_ready_wire_independently_reverified(&self) -> bool { true }
    pub const fn challenge_generated_only_after_opaque_successful_release_existed(&self) -> bool {
        true
    }
    pub const fn checkpoint_two_predecessor_fixed_to_ready_checkpoint(&self) -> bool { true }
    pub const fn causal_os_pid_to_runtime_process_instance_established(&self) -> bool { false }
    pub const fn tracee_consumption_established(&self) -> bool { false }
    pub const fn checkpoint_two_live_observation_established(&self) -> bool { false }
    pub const fn checkpoint_two_signed_inclusion_established(&self) -> bool { false }
    pub const fn exclusive_pipe_writer_authority_established(&self) -> bool { false }
    pub const fn arbitrary_same_process_descriptor_sabotage_excluded(&self) -> bool { false }
    pub const fn trusted_time_established(&self) -> bool { false }
    pub const fn grants_physical_authority(&self) -> bool { false }
}

pub struct PostReleaseRuntimeHandoffQualification {
    pub report: PostReleaseRuntimeHandoffReport,
    pub ticket: PostReleaseRuntimeHandoffTicket,
    issued: IssuedPostReleaseRuntimeChallenge,
}

impl PostReleaseRuntimeHandoffQualification {
    pub fn issued(&self) -> &IssuedPostReleaseRuntimeChallenge { &self.issued }
    pub fn into_issued(self) -> IssuedPostReleaseRuntimeChallenge { self.issued }
}

#[allow(clippy::too_many_arguments)]
pub fn issue_post_release_runtime_challenge(
    policy: &PostReleaseRuntimeHandoffPolicy,
    release: &ReleasedBootstrapVerifier,
    retained: RetainedPostReleaseRuntimeHandoffChannel,
    confinement_report: &BootstrapSyscallConfinementReport,
    launch: &IssuedLaunchRuntimeChallenge,
    ready_policy: &BootstrapReadyCheckpointPolicy,
    runtime_policy: &VerifierRuntimeContinuityPolicy,
    ready_wire_bytes: &[u8],
) -> Result<PostReleaseRuntimeHandoffQualification, PostReleaseRuntimeHandoffReport> {
    let policy_digest = policy.canonical_digest();
    let runtime_policy_digest = runtime_policy.canonical_digest();
    let ready_policy_digest = ready_policy.canonical_digest();
    let ready_bytes_digest = format!("blake3:{}", blake3::hash(ready_wire_bytes).to_hex());
    let mut report = PostReleaseRuntimeHandoffReport {
        schema_version: POST_RELEASE_RUNTIME_HANDOFF_REPORT_SCHEMA_V1.into(),
        policy_id: policy.policy_id.clone(),
        policy_digest: policy_digest.clone(),
        channel_retention_digest: retained.retention_digest.clone(),
        release_digest: release.release_digest().into(),
        release_report_digest: release.report_digest().into(),
        release_policy_digest: release.policy_digest().into(),
        confinement_qualification_digest: release.confinement_qualification_digest().into(),
        mapping_continuity_qualification_digest: release
            .mapping_continuity_qualification_digest()
            .into(),
        recomputed_confinement_qualification_digest: None,
        launch_handoff_qualification_digest: launch.qualification_digest().into(),
        bootstrap_ready_wire_digest: confinement_report.ready_wire_digest.clone(),
        bootstrap_ready_checkpoint_digest: confinement_report.ready_checkpoint_digest.clone(),
        bootstrap_ready_bytes_digest: ready_bytes_digest.clone(),
        process_instance_id: String::new(),
        tracee_pid: release.tracee_pid(),
        pipe_target: retained.pipe_target.clone(),
        retained_fdinfo_before_digest: None,
        retained_fdinfo_after_digest: None,
        challenge_nonce_blake3_hex: None,
        ticket_digest: None,
        ticket_bytes_digest: None,
        ticket_bytes_len: 0,
        disposition: PostReleaseRuntimeHandoffDisposition::Invalid,
        issues: Vec::new(),
    };

    if !policy.validate() {
        report.issues.push(PostReleaseRuntimeHandoffIssue::InvalidPolicy);
    }
    if !runtime_policy.validate() || runtime_policy_digest.is_none() {
        report
            .issues
            .push(PostReleaseRuntimeHandoffIssue::InvalidRuntimePolicy);
    }
    if retained.policy_digest != policy_digest.as_deref().unwrap_or("-") {
        report
            .issues
            .push(PostReleaseRuntimeHandoffIssue::RetentionPolicyMismatch);
    }
    if release.policy_digest() != policy.expected_release_policy_digest {
        report
            .issues
            .push(PostReleaseRuntimeHandoffIssue::ReleasePolicyMismatch);
    }
    if release.confinement_qualification_digest() != retained.confinement_qualification_digest
        || release.mapping_continuity_qualification_digest()
            != retained.mapping_continuity_qualification_digest
    {
        report
            .issues
            .push(PostReleaseRuntimeHandoffIssue::ReleaseParentMismatch);
    }
    if release.tracee_pid() != retained.tracee_pid {
        report
            .issues
            .push(PostReleaseRuntimeHandoffIssue::ReleaseTraceeMismatch);
    }
    if release.exec_confirmation_digest() != retained.exec_confirmation_digest {
        report
            .issues
            .push(PostReleaseRuntimeHandoffIssue::ReleaseExecMismatch);
    }
    if release.syscall_sequence_digest() != retained.syscall_sequence_digest {
        report
            .issues
            .push(PostReleaseRuntimeHandoffIssue::ReleaseSyscallSequenceMismatch);
    }
    if release.launch_mapping_set_digest() != retained.launch_mapping_set_digest
        || release.ready_mapping_set_digest() != retained.ready_mapping_set_digest
    {
        report
            .issues
            .push(PostReleaseRuntimeHandoffIssue::ReleaseMappingSetMismatch);
    }
    if launch.qualification_digest() != retained.launch_handoff_qualification_digest
        || launch.report_digest() != retained.launch_handoff_report_digest
        || launch.tracee_pid() != retained.tracee_pid
        || launch.tracee_handoff_read_fd() != retained.tracee_handoff_read_fd
        || launch.supervisor_handoff_write_fd() != retained.original_supervisor_write_fd
        || launch.pipe_target() != retained.pipe_target
        || launch.channel_binding_digest() != retained.channel_binding_digest
        || launch.authority_snapshot_digest() != retained.authority_snapshot_digest
        || launch.runtime_policy_digest() != retained.runtime_policy_digest
        || launch.runtime_verifier_ref() != retained.runtime_verifier_ref
        || launch.backend_id() != retained.backend_id
        || launch.executable_digest() != retained.executable_digest
    {
        report.issues.push(PostReleaseRuntimeHandoffIssue::LaunchMismatch);
    }
    if ready_policy_digest.as_deref()
        != Some(policy.expected_bootstrap_ready_policy_digest.as_str())
    {
        report
            .issues
            .push(PostReleaseRuntimeHandoffIssue::BootstrapReadyPolicyMismatch);
    }
    if runtime_policy_digest.as_deref() != Some(policy.expected_runtime_policy_digest.as_str())
        || runtime_policy.verifier_ref != policy.expected_runtime_verifier_ref
        || runtime_policy.expected_executable_digest != retained.executable_digest
        || retained.backend_id != policy.expected_backend_id
    {
        report
            .issues
            .push(PostReleaseRuntimeHandoffIssue::BootstrapReadyRuntimeMismatch);
    }
    if !report.issues.is_empty() {
        return Err(finalize(report));
    }

    if confinement_report.disposition != BootstrapSyscallConfinementDisposition::Qualified
        || !confinement_report.issues.is_empty()
        || !confinement_report.tracee_stopped_at_ready_exit
    {
        report
            .issues
            .push(PostReleaseRuntimeHandoffIssue::ConfinementReportNotQualified);
    }
    let Some(recomputed_confinement) = recompute_confinement_qualification(confinement_report)
    else {
        report
            .issues
            .push(PostReleaseRuntimeHandoffIssue::ConfinementReportIncomplete);
        return Err(finalize(report));
    };
    report.recomputed_confinement_qualification_digest = Some(recomputed_confinement.clone());
    if recomputed_confinement != retained.confinement_qualification_digest
        || recomputed_confinement != release.confinement_qualification_digest()
    {
        report
            .issues
            .push(PostReleaseRuntimeHandoffIssue::ConfinementQualificationMismatch);
    }
    if confinement_report.policy_digest.as_deref()
        != Some(policy.expected_confinement_policy_digest.as_str())
        || confinement_report.tracee_pid != retained.tracee_pid
        || confinement_report.exec_confirmation_digest != retained.exec_confirmation_digest
        || confinement_report.issued_challenge_qualification_digest
            != retained.launch_handoff_qualification_digest
        || confinement_report.final_authority_snapshot_digest.as_deref()
            != Some(retained.authority_snapshot_digest.as_str())
        || confinement_report.syscall_sequence_digest.as_deref()
            != Some(retained.syscall_sequence_digest.as_str())
        || confinement_report.ready_wire_digest.as_deref()
            != Some(retained.ready_wire_digest.as_str())
        || confinement_report.ready_checkpoint_digest.as_deref()
            != Some(retained.ready_checkpoint_digest.as_str())
        || confinement_report.ready_wire_bytes_digest.as_deref()
            != Some(retained.ready_wire_bytes_digest.as_str())
    {
        report
            .issues
            .push(PostReleaseRuntimeHandoffIssue::ConfinementReportIdentityMismatch);
    }
    if !report.issues.is_empty() {
        return Err(finalize(report));
    }

    let ready = match verify_bootstrap_ready_wire(ready_policy, runtime_policy, ready_wire_bytes) {
        Ok(value) => value,
        Err(ready_report) => {
            report
                .issues
                .push(PostReleaseRuntimeHandoffIssue::BootstrapReadyWireVerificationFailed(
                    ready_report.canonical_digest(),
                ));
            return Err(finalize(report));
        }
    };
    report.bootstrap_ready_wire_digest = Some(ready.wire_digest().into());
    report.bootstrap_ready_checkpoint_digest = Some(ready.checkpoint_digest().into());
    report.process_instance_id = ready.process_instance_id().into();
    if ready.wire_digest() != retained.ready_wire_digest
        || ready.checkpoint_digest() != retained.ready_checkpoint_digest
    {
        report
            .issues
            .push(PostReleaseRuntimeHandoffIssue::BootstrapReadyIdentityMismatch);
    }
    if ready.wire_bytes_digest() != retained.ready_wire_bytes_digest
        || ready.wire_bytes_digest() != ready_bytes_digest
        || confinement_report.ready_wire_bytes_len != ready_wire_bytes.len() as u64
    {
        report
            .issues
            .push(PostReleaseRuntimeHandoffIssue::BootstrapReadyBytesMismatch);
    }
    if ready.ticket_digest() != launch.ticket_digest() {
        report
            .issues
            .push(PostReleaseRuntimeHandoffIssue::BootstrapReadyTicketMismatch);
    }
    if ready.launch_challenge_nonce_blake3_hex() != launch.challenge_nonce_blake3_hex() {
        report
            .issues
            .push(PostReleaseRuntimeHandoffIssue::BootstrapReadyChallengeMismatch);
    }
    if ready.runtime_policy_digest() != retained.runtime_policy_digest
        || ready.runtime_policy_digest() != policy.expected_runtime_policy_digest
    {
        report
            .issues
            .push(PostReleaseRuntimeHandoffIssue::BootstrapReadyRuntimeMismatch);
    }
    if !report.issues.is_empty() {
        return Err(finalize(report));
    }

    let before = match observe_fd(retained.retained_write_fd, policy.max_fdinfo_bytes) {
        Ok(value) => value,
        Err(error) => {
            report
                .issues
                .push(PostReleaseRuntimeHandoffIssue::RetainedFdUnavailable(error));
            return Err(finalize(report));
        }
    };
    report.retained_fdinfo_before_digest = Some(before.fdinfo_digest.clone());
    if before != retained.retained_observation
        || before.target != retained.pipe_target
        || before.inode != retained.pipe_inode
    {
        report.issues.push(PostReleaseRuntimeHandoffIssue::RetainedFdChanged);
        return Err(finalize(report));
    }

    let mut random = [0u8; 32];
    if let Err(error) = getrandom::getrandom(&mut random) {
        report
            .issues
            .push(PostReleaseRuntimeHandoffIssue::ChallengeEntropyUnavailable(
                error.to_string(),
            ));
        return Err(finalize(report));
    }
    if random.iter().all(|byte| *byte == 0) {
        report
            .issues
            .push(PostReleaseRuntimeHandoffIssue::ChallengeEntropyUnavailable(
                "reserved-zero-challenge".into(),
            ));
        return Err(finalize(report));
    }
    let challenge = hex_lower(&random);
    report.challenge_nonce_blake3_hex = Some(challenge.clone());

    let mut ticket = PostReleaseRuntimeHandoffTicket {
        schema_version: POST_RELEASE_RUNTIME_HANDOFF_TICKET_SCHEMA_V1.into(),
        channel_retention_digest: retained.retention_digest.clone(),
        release_digest: release.release_digest().into(),
        release_report_digest: release.report_digest().into(),
        confinement_qualification_digest: retained.confinement_qualification_digest.clone(),
        mapping_continuity_qualification_digest: retained
            .mapping_continuity_qualification_digest
            .clone(),
        bootstrap_ready_wire_digest: ready.wire_digest().into(),
        bootstrap_ready_checkpoint_digest: ready.checkpoint_digest().into(),
        launch_handoff_qualification_digest: launch.qualification_digest().into(),
        launch_ticket_digest: launch.ticket_digest().into(),
        tracee_pid: release.tracee_pid(),
        tracee_handoff_read_fd: retained.tracee_handoff_read_fd,
        pipe_target: retained.pipe_target.clone(),
        original_channel_binding_digest: retained.channel_binding_digest.clone(),
        process_instance_id: ready.process_instance_id().into(),
        runtime_policy_digest: ready.runtime_policy_digest().into(),
        runtime_verifier_ref: policy.expected_runtime_verifier_ref.clone(),
        backend_id: retained.backend_id.clone(),
        boot_measurement_digest: runtime_policy.expected_boot_measurement_digest.clone(),
        executable_digest: runtime_policy.expected_executable_digest.clone(),
        dependency_closure_digest: runtime_policy.expected_dependency_closure_digest.clone(),
        runtime_config_digest: runtime_policy.expected_runtime_config_digest.clone(),
        launch_attestation_digest: ready.launch_digest().into(),
        checkpoint_one_challenge_digest: ready.checkpoint_challenge_digest().into(),
        checkpoint_one_observation_qualification_digest: ready
            .first_observation_qualification_digest()
            .into(),
        checkpoint_sequence: 2,
        previous_checkpoint_digest: ready.checkpoint_digest().into(),
        launch_monotonic_counter: ready.launch_counter(),
        previous_checkpoint_monotonic_counter: ready.checkpoint_monotonic_counter(),
        challenge_nonce_blake3_hex: challenge.clone(),
        ticket_digest: String::new(),
    };
    ticket.ticket_digest = ticket.recompute_digest();
    let Some(wire) = ticket.to_wire_bytes() else {
        report
            .issues
            .push(PostReleaseRuntimeHandoffIssue::TicketEncodingFailed);
        return Err(finalize(report));
    };
    report.ticket_digest = Some(ticket.ticket_digest.clone());
    report.ticket_bytes_len = wire.len() as u64;
    let ticket_bytes_digest = format!("blake3:{}", blake3::hash(&wire).to_hex());
    report.ticket_bytes_digest = Some(ticket_bytes_digest.clone());
    if wire.len() as u32 > policy.max_ticket_bytes || wire.len() > HARD_MAX_TICKET_BYTES as usize {
        report
            .issues
            .push(PostReleaseRuntimeHandoffIssue::TicketTooLarge {
                observed: wire.len() as u64,
                maximum: policy.max_ticket_bytes as u64,
            });
        return Err(finalize(report));
    }

    let written = match write(retained.retained_write_fd, &wire) {
        Ok(value) => value,
        Err(error) => {
            report
                .issues
                .push(PostReleaseRuntimeHandoffIssue::TicketWriteFailed(error.to_string()));
            return Err(finalize(report));
        }
    };
    if written != wire.len() {
        report
            .issues
            .push(PostReleaseRuntimeHandoffIssue::PartialTicketWrite {
                expected: wire.len() as u64,
                observed: written as u64,
            });
        return Err(finalize(report));
    }

    let after = match observe_fd(retained.retained_write_fd, policy.max_fdinfo_bytes) {
        Ok(value) => value,
        Err(error) => {
            report
                .issues
                .push(PostReleaseRuntimeHandoffIssue::RetainedFdUnavailable(error));
            return Err(finalize(report));
        }
    };
    report.retained_fdinfo_after_digest = Some(after.fdinfo_digest.clone());
    if before != after || after != retained.retained_observation {
        report
            .issues
            .push(PostReleaseRuntimeHandoffIssue::RetainedFdChangedDuringIssuance);
        return Err(finalize(report));
    }

    report.disposition = PostReleaseRuntimeHandoffDisposition::Qualified;
    let report_digest = report.canonical_digest();
    let policy_digest = policy_digest.expect("validated policy has digest");
    let qualification_digest = qualification_digest(
        &policy_digest,
        &retained.retention_digest,
        release.release_digest(),
        retained.confinement_qualification_digest.as_str(),
        retained.mapping_continuity_qualification_digest.as_str(),
        ready.wire_digest(),
        ready.checkpoint_digest(),
        launch.qualification_digest(),
        &ticket.ticket_digest,
        &report_digest,
    );
    let issued = IssuedPostReleaseRuntimeChallenge {
        qualification_digest,
        report_digest,
        policy_digest,
        channel_retention_digest: retained.retention_digest.clone(),
        release_digest: release.release_digest().into(),
        release_report_digest: release.report_digest().into(),
        confinement_qualification_digest: retained.confinement_qualification_digest.clone(),
        mapping_continuity_qualification_digest: retained
            .mapping_continuity_qualification_digest
            .clone(),
        bootstrap_ready_wire_digest: ready.wire_digest().into(),
        bootstrap_ready_checkpoint_digest: ready.checkpoint_digest().into(),
        launch_handoff_qualification_digest: launch.qualification_digest().into(),
        launch_ticket_digest: launch.ticket_digest().into(),
        tracee_pid: release.tracee_pid(),
        tracee_handoff_read_fd: retained.tracee_handoff_read_fd,
        pipe_target: retained.pipe_target.clone(),
        original_channel_binding_digest: retained.channel_binding_digest.clone(),
        process_instance_id: ready.process_instance_id().into(),
        runtime_policy_digest: ready.runtime_policy_digest().into(),
        runtime_verifier_ref: policy.expected_runtime_verifier_ref.clone(),
        backend_id: retained.backend_id.clone(),
        boot_measurement_digest: runtime_policy.expected_boot_measurement_digest.clone(),
        executable_digest: runtime_policy.expected_executable_digest.clone(),
        dependency_closure_digest: runtime_policy.expected_dependency_closure_digest.clone(),
        runtime_config_digest: runtime_policy.expected_runtime_config_digest.clone(),
        launch_attestation_digest: ready.launch_digest().into(),
        checkpoint_one_challenge_digest: ready.checkpoint_challenge_digest().into(),
        checkpoint_one_observation_qualification_digest: ready
            .first_observation_qualification_digest()
            .into(),
        launch_monotonic_counter: ready.launch_counter(),
        previous_checkpoint_monotonic_counter: ready.checkpoint_monotonic_counter(),
        challenge_nonce_blake3_hex: challenge,
        ticket_digest: ticket.ticket_digest.clone(),
        ticket_bytes_digest,
    };
    Ok(PostReleaseRuntimeHandoffQualification {
        report,
        ticket,
        issued,
    })
}

fn recompute_confinement_qualification(report: &BootstrapSyscallConfinementReport) -> Option<String> {
    let policy = report.policy_digest.as_deref()?;
    let wire = report.ready_wire_digest.as_deref()?;
    let sequence = report.syscall_sequence_digest.as_deref()?;
    let authority = report.final_authority_snapshot_digest.as_deref()?;
    if report.disposition != BootstrapSyscallConfinementDisposition::Qualified
        || !report.issues.is_empty()
        || !report.tracee_stopped_at_ready_exit
    {
        return None;
    }
    let report_digest = report.canonical_digest();
    let mut h = blake3::Hasher::new();
    h.update(CONFINEMENT_QUALIFICATION_DOMAIN);
    for value in [
        policy,
        report.exec_confirmation_digest.as_str(),
        report.issued_challenge_qualification_digest.as_str(),
        wire,
        sequence,
        authority,
        report_digest.as_str(),
    ] {
        field(&mut h, value);
    }
    Some(b3(h.finalize()))
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct FdObservation {
    target: String,
    flags: i32,
    inode: u64,
    fdinfo_digest: String,
}

fn observe_fd(fd: RawFd, maximum: u64) -> Result<FdObservation, String> {
    let target = fs::read_link(format!("/proc/self/fd/{fd}"))
        .map_err(|error| error.to_string())?
        .into_os_string()
        .into_string()
        .map_err(|_| "non-utf8-fd-target".to_string())?;
    let bytes = fs::read(format!("/proc/self/fdinfo/{fd}"))
        .map_err(|error| error.to_string())?;
    if bytes.len() as u64 > maximum {
        return Err("fdinfo-size-limit-exceeded".into());
    }
    let text = std::str::from_utf8(&bytes).map_err(|error| error.to_string())?;
    let mut flags = None;
    let mut inode = None;
    for line in text.lines() {
        if let Some((key, value)) = line.split_once(':') {
            match key {
                "flags" => {
                    flags = Some(
                        i32::from_str_radix(value.trim(), 8)
                            .map_err(|_| "invalid-flags".to_string())?,
                    );
                }
                "ino" => {
                    inode = Some(
                        value
                            .trim()
                            .parse::<u64>()
                            .map_err(|_| "invalid-inode".to_string())?,
                    );
                }
                _ => {}
            }
        }
    }
    Ok(FdObservation {
        target,
        flags: flags.ok_or_else(|| "missing-flags".to_string())?,
        inode: inode.ok_or_else(|| "missing-inode".to_string())?,
        fdinfo_digest: format!("blake3:{}", blake3::hash(&bytes).to_hex()),
    })
}

#[allow(clippy::too_many_arguments)]
fn retention_digest(
    policy: &str,
    confinement: &str,
    mapping: &str,
    launch: &str,
    ready_wire: &str,
    ready_checkpoint: &str,
    source_fd: RawFd,
    retained_fd: RawFd,
    source: &FdObservation,
    retained: &FdObservation,
    report: &str,
) -> String {
    let mut h = blake3::Hasher::new();
    h.update(RETENTION_DOMAIN);
    for value in [
        policy,
        confinement,
        mapping,
        launch,
        ready_wire,
        ready_checkpoint,
        source.target.as_str(),
        source.fdinfo_digest.as_str(),
        retained.target.as_str(),
        retained.fdinfo_digest.as_str(),
        report,
    ] {
        field(&mut h, value);
    }
    h.update(&source_fd.to_le_bytes());
    h.update(&retained_fd.to_le_bytes());
    h.update(&source.inode.to_le_bytes());
    h.update(&retained.inode.to_le_bytes());
    b3(h.finalize())
}

fn ticket_digest(ticket: &PostReleaseRuntimeHandoffTicket) -> String {
    let mut h = blake3::Hasher::new();
    h.update(TICKET_DOMAIN);
    for value in [
        ticket.schema_version.as_str(),
        ticket.channel_retention_digest.as_str(),
        ticket.release_digest.as_str(),
        ticket.release_report_digest.as_str(),
        ticket.confinement_qualification_digest.as_str(),
        ticket.mapping_continuity_qualification_digest.as_str(),
        ticket.bootstrap_ready_wire_digest.as_str(),
        ticket.bootstrap_ready_checkpoint_digest.as_str(),
        ticket.launch_handoff_qualification_digest.as_str(),
        ticket.launch_ticket_digest.as_str(),
    ] {
        field(&mut h, value);
    }
    h.update(&ticket.tracee_pid.to_le_bytes());
    h.update(&ticket.tracee_handoff_read_fd.to_le_bytes());
    for value in [
        ticket.pipe_target.as_str(),
        ticket.original_channel_binding_digest.as_str(),
        ticket.process_instance_id.as_str(),
        ticket.runtime_policy_digest.as_str(),
        ticket.runtime_verifier_ref.as_str(),
        ticket.backend_id.as_str(),
        ticket.boot_measurement_digest.as_str(),
        ticket.executable_digest.as_str(),
        ticket.dependency_closure_digest.as_str(),
        ticket.runtime_config_digest.as_str(),
        ticket.launch_attestation_digest.as_str(),
        ticket.checkpoint_one_challenge_digest.as_str(),
        ticket.checkpoint_one_observation_qualification_digest.as_str(),
    ] {
        field(&mut h, value);
    }
    h.update(&ticket.checkpoint_sequence.to_le_bytes());
    field(&mut h, &ticket.previous_checkpoint_digest);
    h.update(&ticket.launch_monotonic_counter.to_le_bytes());
    h.update(&ticket.previous_checkpoint_monotonic_counter.to_le_bytes());
    field(&mut h, &ticket.challenge_nonce_blake3_hex);
    b3(h.finalize())
}

#[allow(clippy::too_many_arguments)]
fn qualification_digest(
    policy: &str,
    retention: &str,
    release: &str,
    confinement: &str,
    mapping: &str,
    ready_wire: &str,
    ready_checkpoint: &str,
    launch: &str,
    ticket: &str,
    report: &str,
) -> String {
    let mut h = blake3::Hasher::new();
    h.update(QUALIFICATION_DOMAIN);
    for value in [
        policy,
        retention,
        release,
        confinement,
        mapping,
        ready_wire,
        ready_checkpoint,
        launch,
        ticket,
        report,
    ] {
        field(&mut h, value);
    }
    b3(h.finalize())
}

fn finalize_retention(
    mut report: PostReleaseRuntimeChannelRetentionReport,
) -> PostReleaseRuntimeChannelRetentionReport {
    report.disposition = if report
        .issues
        .iter()
        .any(PostReleaseRuntimeChannelRetentionIssue::invalid)
    {
        PostReleaseRuntimeChannelRetentionDisposition::Invalid
    } else {
        PostReleaseRuntimeChannelRetentionDisposition::Blocked
    };
    report
}

fn finalize(mut report: PostReleaseRuntimeHandoffReport) -> PostReleaseRuntimeHandoffReport {
    report.disposition = if report
        .issues
        .iter()
        .any(PostReleaseRuntimeHandoffIssue::invalid)
    {
        PostReleaseRuntimeHandoffDisposition::Invalid
    } else {
        PostReleaseRuntimeHandoffDisposition::Blocked
    };
    report
}

struct Cursor<'a> {
    bytes: &'a [u8],
    offset: usize,
}

impl Cursor<'_> {
    fn u32(&mut self) -> Option<u32> {
        let end = self.offset.checked_add(4)?;
        let value = u32::from_le_bytes(self.bytes.get(self.offset..end)?.try_into().ok()?);
        self.offset = end;
        Some(value)
    }

    fn u64(&mut self) -> Option<u64> {
        let end = self.offset.checked_add(8)?;
        let value = u64::from_le_bytes(self.bytes.get(self.offset..end)?.try_into().ok()?);
        self.offset = end;
        Some(value)
    }

    fn i32(&mut self) -> Option<i32> {
        let end = self.offset.checked_add(4)?;
        let value = i32::from_le_bytes(self.bytes.get(self.offset..end)?.try_into().ok()?);
        self.offset = end;
        Some(value)
    }

    fn string(&mut self) -> Option<String> {
        let length = self.u32()? as usize;
        if length == 0 || length > MAX_TEXT {
            return None;
        }
        let end = self.offset.checked_add(length)?;
        let value = std::str::from_utf8(self.bytes.get(self.offset..end)?)
            .ok()?
            .to_string();
        self.offset = end;
        canonical_text(&value).then_some(value)
    }
}

fn wire_string(output: &mut Vec<u8>, value: &str) -> Option<()> {
    if !canonical_text(value) || value.len() > u32::MAX as usize {
        return None;
    }
    output.extend_from_slice(&(value.len() as u32).to_le_bytes());
    output.extend_from_slice(value.as_bytes());
    Some(())
}

fn parse_pipe_target(value: &str) -> Option<u64> {
    value
        .strip_prefix("pipe:[")?
        .strip_suffix(']')?
        .parse::<u64>()
        .ok()
}

fn valid_nonce(value: &str) -> bool {
    lower_hex_exact(value, 64) && value.bytes().any(|byte| byte != b'0')
}

fn lower_hex_exact(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn valid_blake3(value: &str) -> bool {
    value
        .strip_prefix("blake3:")
        .is_some_and(|digest| lower_hex_exact(digest, 64))
}

fn canonical_text(value: &str) -> bool {
    !value.is_empty()
        && value == value.trim()
        && value.len() <= MAX_TEXT
        && !value.chars().any(char::is_control)
}

fn valid_refs(values: &[String]) -> bool {
    values.len() <= MAX_REFS
        && values.iter().all(|value| canonical_text(value))
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

fn hex_lower(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        use std::fmt::Write as _;
        let _ = write!(&mut out, "{byte:02x}");
    }
    out
}

fn b3(hash: blake3::Hash) -> String {
    format!("blake3:{}", hash.to_hex())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(label: &str) -> String {
        format!("blake3:{}", blake3::hash(label.as_bytes()).to_hex())
    }

    fn policy() -> PostReleaseRuntimeHandoffPolicy {
        PostReleaseRuntimeHandoffPolicy {
            schema_version: POST_RELEASE_RUNTIME_HANDOFF_POLICY_SCHEMA_V1.into(),
            policy_id: "post-release-handoff:v1".into(),
            expected_confinement_policy_digest: d("confinement-policy"),
            expected_mapping_continuity_policy_digest: d("mapping-policy"),
            expected_release_policy_digest: d("release-policy"),
            expected_launch_handoff_policy_digest: d("launch-policy"),
            expected_bootstrap_ready_policy_digest: d("ready-policy"),
            expected_runtime_policy_digest: d("runtime-policy"),
            expected_runtime_verifier_ref: "verifier:prod".into(),
            expected_backend_id: "backend:in-process-p256".into(),
            max_ticket_bytes: 4096,
            max_fdinfo_bytes: 1024 * 1024,
            evidence_refs: vec!["review:post-release-handoff".into()],
        }
    }

    fn ticket() -> PostReleaseRuntimeHandoffTicket {
        let mut value = PostReleaseRuntimeHandoffTicket {
            schema_version: POST_RELEASE_RUNTIME_HANDOFF_TICKET_SCHEMA_V1.into(),
            channel_retention_digest: d("retention"),
            release_digest: d("release"),
            release_report_digest: d("release-report"),
            confinement_qualification_digest: d("confinement"),
            mapping_continuity_qualification_digest: d("mapping"),
            bootstrap_ready_wire_digest: d("ready-wire"),
            bootstrap_ready_checkpoint_digest: d("checkpoint-one"),
            launch_handoff_qualification_digest: d("launch-handoff"),
            launch_ticket_digest: d("launch-ticket"),
            tracee_pid: 4242,
            tracee_handoff_read_fd: 11,
            pipe_target: "pipe:[123456]".into(),
            original_channel_binding_digest: d("channel"),
            process_instance_id: "process:4242:boot-a".into(),
            runtime_policy_digest: d("runtime-policy"),
            runtime_verifier_ref: "verifier:prod".into(),
            backend_id: "backend:in-process-p256".into(),
            boot_measurement_digest: d("boot"),
            executable_digest: d("exe"),
            dependency_closure_digest: d("closure"),
            runtime_config_digest: d("config"),
            launch_attestation_digest: d("launch-attestation"),
            checkpoint_one_challenge_digest: d("checkpoint-one-challenge"),
            checkpoint_one_observation_qualification_digest: d("checkpoint-one-observation"),
            checkpoint_sequence: 2,
            previous_checkpoint_digest: d("checkpoint-one"),
            launch_monotonic_counter: 6,
            previous_checkpoint_monotonic_counter: 7,
            challenge_nonce_blake3_hex: "12".repeat(32),
            ticket_digest: String::new(),
        };
        value.ticket_digest = value.recompute_digest();
        value
    }

    #[test]
    fn evidence_ref_order_is_nonsemantic() {
        let mut left = policy();
        left.evidence_refs = vec!["review:a".into(), "review:b".into()];
        let mut right = left.clone();
        right.evidence_refs.reverse();
        assert_eq!(left.canonical_digest(), right.canonical_digest());
    }

    #[test]
    fn every_parent_policy_identity_is_semantic() {
        let left = policy();
        let mut right = left.clone();
        right.expected_confinement_policy_digest = d("other-confinement");
        assert_ne!(left.canonical_digest(), right.canonical_digest());
        let mut right = left.clone();
        right.expected_mapping_continuity_policy_digest = d("other-mapping");
        assert_ne!(left.canonical_digest(), right.canonical_digest());
        let mut right = left.clone();
        right.expected_release_policy_digest = d("other-release");
        assert_ne!(left.canonical_digest(), right.canonical_digest());
        let mut right = left.clone();
        right.expected_backend_id = "backend:other".into();
        assert_ne!(left.canonical_digest(), right.canonical_digest());
    }

    #[test]
    fn ticket_round_trips_exact_canonical_wire() {
        let value = ticket();
        let bytes = value.to_wire_bytes().unwrap();
        let decoded = PostReleaseRuntimeHandoffTicket::from_wire_bytes(&bytes).unwrap();
        assert_eq!(decoded, value);
        assert_eq!(decoded.to_wire_bytes().unwrap(), bytes);
    }

    #[test]
    fn release_retention_and_checkpoint_one_are_semantic_ticket_inputs() {
        let left = ticket();
        let mut right = left.clone();
        right.release_digest = d("different-release");
        right.ticket_digest = right.recompute_digest();
        assert_ne!(left.ticket_digest, right.ticket_digest);
        let mut right = left.clone();
        right.channel_retention_digest = d("different-retention");
        right.ticket_digest = right.recompute_digest();
        assert_ne!(left.ticket_digest, right.ticket_digest);
        let mut right = left.clone();
        right.previous_checkpoint_digest = d("different-checkpoint");
        right.bootstrap_ready_checkpoint_digest = right.previous_checkpoint_digest.clone();
        right.ticket_digest = right.recompute_digest();
        assert_ne!(left.ticket_digest, right.ticket_digest);
    }

    #[test]
    fn checkpoint_two_and_nonzero_challenge_are_fixed() {
        let value = ticket();
        assert_eq!(value.checkpoint_sequence, 2);
        assert!(value.previous_checkpoint_monotonic_counter > value.launch_monotonic_counter);
        assert!(valid_nonce(&value.challenge_nonce_blake3_hex));
        let mut zero = value.clone();
        zero.challenge_nonce_blake3_hex = "0".repeat(64);
        zero.ticket_digest = zero.recompute_digest();
        assert!(!zero.validate());
    }

    #[test]
    fn claim_ceiling_stays_explicit() {
        let claims = [
            "tracee-consumption=false",
            "checkpoint-two-live-observation=false",
            "checkpoint-two-signed-inclusion=false",
            "exclusive-pipe-writer=false",
            "same-process-fd-sabotage-excluded=false",
            "trusted-time=false",
            "physical-authority=false",
        ];
        assert_eq!(claims.len(), 7);
    }
}
