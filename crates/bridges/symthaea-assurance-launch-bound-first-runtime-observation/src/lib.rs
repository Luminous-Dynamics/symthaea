// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Tracee-side launch handoff consumption before checkpoint-one mapped observation.
//!
//! The constructor reads one canonical handoff ticket from a reviewed inherited
//! pipe fd, proves that the current Linux PID and runtime identity match the
//! ticket, constructs checkpoint sequence 1 with no predecessor using the
//! ticket challenge, and only then invokes the live fresh mapped-runtime
//! observer. Supervisor-issued-ticket authenticity remains a later composition
//! theorem because exclusive pipe-writer authority is not established here.

#![cfg(target_os = "linux")]
#![deny(unsafe_code)]

use nix::{fcntl::OFlag, unistd::read};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeSet,
    fs,
    os::unix::io::RawFd,
    path::PathBuf,
};
use symthaea_assurance_closure_backed_in_process_continuous::
    ClosureBackedInProcessContinuousVerifierExecution;
use symthaea_assurance_fresh_mapped_runtime_continuity::{
    observe_fresh_mapped_runtime_for_checkpoint, FreshMappedRuntimeObservationQualification,
    FreshMappedRuntimePolicy, MappedRuntimeCheckpointChallenge,
};
use symthaea_assurance_launch_runtime_handoff_channel::LaunchRuntimeHandoffTicket;
use symthaea_assurance_nix_bound_in_process_runtime::NixBoundInProcessRuntimePolicy;
use symthaea_assurance_nix_runtime_closure::NixRuntimeClosureQualification;
use symthaea_assurance_observed_mapped_nix_executable_runtime::MappedExecutableRuntimePolicy;

pub const LAUNCH_BOUND_FIRST_OBSERVATION_POLICY_SCHEMA_V1: &str =
    "symthaea.assurance.launch-bound-first-runtime-observation-policy.v1";
pub const LAUNCH_BOUND_FIRST_OBSERVATION_REPORT_SCHEMA_V1: &str =
    "symthaea.assurance.launch-bound-first-runtime-observation-report.v1";

const POLICY_DOMAIN: &[u8] =
    b"symthaea.assurance.launch-bound-first-runtime-observation-policy.digest.v1\0";
const PIPE_OBSERVATION_DOMAIN: &[u8] =
    b"symthaea.assurance.launch-bound-first-runtime-pipe-observation.digest.v1\0";
const REPORT_DOMAIN: &[u8] =
    b"symthaea.assurance.launch-bound-first-runtime-observation-report.digest.v1\0";
const QUALIFICATION_DOMAIN: &[u8] =
    b"symthaea.assurance.launch-bound-first-runtime-observation-qualification.digest.v1\0";
const WIRE_MAGIC: &[u8] = b"SYMT-LR-HANDOFF-V1\0";
const STRING_FIELDS_BEFORE_IDS: usize = 6;
const STRING_FIELDS_AFTER_IDS: usize = 8;
const MAX_TEXT: usize = 4096;
const MAX_REFS: usize = 128;
const HARD_MAX_TICKET_BYTES: u32 = 4096;
const HARD_MAX_FDINFO_BYTES: u64 = 4 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchBoundFirstObservationPolicy {
    pub schema_version: String,
    pub policy_id: String,
    pub expected_fresh_mapped_policy_digest: String,
    pub expected_closure_backed_policy_digest: String,
    pub expected_nix_binding_policy_digest: String,
    pub expected_mapped_runtime_policy_digest: String,
    pub expected_runtime_policy_digest: String,
    pub expected_runtime_verifier_ref: String,
    pub expected_backend_id: String,
    pub handoff_read_fd: u32,
    pub max_ticket_bytes: u32,
    pub max_fdinfo_bytes: u64,
    pub evidence_refs: Vec<String>,
}

impl LaunchBoundFirstObservationPolicy {
    pub fn validate(&self) -> bool {
        self.schema_version == LAUNCH_BOUND_FIRST_OBSERVATION_POLICY_SCHEMA_V1
            && canonical_text(&self.policy_id)
            && valid_blake3(&self.expected_fresh_mapped_policy_digest)
            && valid_blake3(&self.expected_closure_backed_policy_digest)
            && valid_blake3(&self.expected_nix_binding_policy_digest)
            && valid_blake3(&self.expected_mapped_runtime_policy_digest)
            && valid_blake3(&self.expected_runtime_policy_digest)
            && canonical_text(&self.expected_runtime_verifier_ref)
            && canonical_text(&self.expected_backend_id)
            && self.handoff_read_fd <= 1_048_576
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
            self.expected_fresh_mapped_policy_digest.as_str(),
            self.expected_closure_backed_policy_digest.as_str(),
            self.expected_nix_binding_policy_digest.as_str(),
            self.expected_mapped_runtime_policy_digest.as_str(),
            self.expected_runtime_policy_digest.as_str(),
            self.expected_runtime_verifier_ref.as_str(),
            self.expected_backend_id.as_str(),
        ] {
            field(&mut h, value);
        }
        h.update(&self.handoff_read_fd.to_le_bytes());
        h.update(&self.max_ticket_bytes.to_le_bytes());
        h.update(&self.max_fdinfo_bytes.to_le_bytes());
        sorted_strings(&mut h, &self.evidence_refs);
        Some(format!("blake3:{}", h.finalize().to_hex()))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LaunchBoundFirstObservationIssue {
    InvalidPolicy,
    InvalidCheckpointCounter,
    FreshMappedPolicyMismatch,
    ClosureBackedPolicyMismatch,
    NixBindingPolicyMismatch,
    MappedRuntimePolicyMismatch,
    RuntimePolicyMismatch,
    RuntimeVerifierMismatch,
    BackendMismatch,
    HandoffFdInvalid,
    HandoffFdUnavailable(String),
    HandoffFdTargetNotPipe,
    HandoffFdNotReadOnly,
    HandoffFdCloseOnExecUnexpected,
    HandoffFdNonBlocking,
    HandoffFdInfoInconsistent,
    TicketReadFailed(String),
    TicketTooLarge { observed: u64, maximum: u64 },
    TicketDecodeFailed,
    TicketReadFdMismatch,
    TicketPipeTargetMismatch,
    TicketPidMismatch { expected: i32, observed: i32 },
    TicketRuntimePolicyMismatch,
    TicketRuntimeVerifierMismatch,
    TicketBackendMismatch,
    TicketExecutableDigestMismatch,
    HandoffFdChangedDuringConsumption,
    ChallengeConstructionFailed,
    FreshObservationFailed(String),
    FreshChallengeMismatch,
    FreshSequenceMismatch,
    FreshPredecessorMismatch,
    FreshProcessMismatch,
    FreshRuntimePolicyMismatch,
    FreshVerifierMismatch,
    FreshBackendMismatch,
    FreshExecutableMismatch,
}

impl LaunchBoundFirstObservationIssue {
    fn invalid(&self) -> bool {
        matches!(
            self,
            Self::InvalidPolicy | Self::InvalidCheckpointCounter | Self::HandoffFdInvalid
        )
    }

    fn code(&self) -> String {
        match self {
            Self::InvalidPolicy => "invalid-policy".into(),
            Self::InvalidCheckpointCounter => "invalid-checkpoint-counter".into(),
            Self::FreshMappedPolicyMismatch => "fresh-mapped-policy-mismatch".into(),
            Self::ClosureBackedPolicyMismatch => "closure-backed-policy-mismatch".into(),
            Self::NixBindingPolicyMismatch => "nix-binding-policy-mismatch".into(),
            Self::MappedRuntimePolicyMismatch => "mapped-runtime-policy-mismatch".into(),
            Self::RuntimePolicyMismatch => "runtime-policy-mismatch".into(),
            Self::RuntimeVerifierMismatch => "runtime-verifier-mismatch".into(),
            Self::BackendMismatch => "backend-mismatch".into(),
            Self::HandoffFdInvalid => "handoff-fd-invalid".into(),
            Self::HandoffFdUnavailable(v) => format!("handoff-fd-unavailable:{v}"),
            Self::HandoffFdTargetNotPipe => "handoff-fd-target-not-pipe".into(),
            Self::HandoffFdNotReadOnly => "handoff-fd-not-read-only".into(),
            Self::HandoffFdCloseOnExecUnexpected => "handoff-fd-close-on-exec-unexpected".into(),
            Self::HandoffFdNonBlocking => "handoff-fd-nonblocking".into(),
            Self::HandoffFdInfoInconsistent => "handoff-fdinfo-inconsistent".into(),
            Self::TicketReadFailed(v) => format!("ticket-read-failed:{v}"),
            Self::TicketTooLarge { observed, maximum } => {
                format!("ticket-too-large:{observed}:{maximum}")
            }
            Self::TicketDecodeFailed => "ticket-decode-failed".into(),
            Self::TicketReadFdMismatch => "ticket-read-fd-mismatch".into(),
            Self::TicketPipeTargetMismatch => "ticket-pipe-target-mismatch".into(),
            Self::TicketPidMismatch { expected, observed } => {
                format!("ticket-pid-mismatch:{expected}:{observed}")
            }
            Self::TicketRuntimePolicyMismatch => "ticket-runtime-policy-mismatch".into(),
            Self::TicketRuntimeVerifierMismatch => "ticket-runtime-verifier-mismatch".into(),
            Self::TicketBackendMismatch => "ticket-backend-mismatch".into(),
            Self::TicketExecutableDigestMismatch => "ticket-executable-digest-mismatch".into(),
            Self::HandoffFdChangedDuringConsumption => "handoff-fd-changed-during-consumption".into(),
            Self::ChallengeConstructionFailed => "challenge-construction-failed".into(),
            Self::FreshObservationFailed(v) => format!("fresh-observation-failed:{v}"),
            Self::FreshChallengeMismatch => "fresh-challenge-mismatch".into(),
            Self::FreshSequenceMismatch => "fresh-sequence-mismatch".into(),
            Self::FreshPredecessorMismatch => "fresh-predecessor-mismatch".into(),
            Self::FreshProcessMismatch => "fresh-process-mismatch".into(),
            Self::FreshRuntimePolicyMismatch => "fresh-runtime-policy-mismatch".into(),
            Self::FreshVerifierMismatch => "fresh-verifier-mismatch".into(),
            Self::FreshBackendMismatch => "fresh-backend-mismatch".into(),
            Self::FreshExecutableMismatch => "fresh-executable-mismatch".into(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LaunchBoundFirstObservationDisposition {
    Invalid,
    Blocked,
    Qualified,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchBoundFirstObservationReport {
    pub schema_version: String,
    pub policy_id: String,
    pub policy_digest: Option<String>,
    pub current_pid: i32,
    pub handoff_read_fd: u32,
    pub pipe_target: Option<String>,
    pub pre_read_fd_observation_digest: Option<String>,
    pub post_read_fd_observation_digest: Option<String>,
    pub ticket_digest: Option<String>,
    pub ticket_bytes_digest: Option<String>,
    pub ticket_bytes_len: u64,
    pub exec_confirmation_digest: Option<String>,
    pub critical_authority_qualification_digest: Option<String>,
    pub authority_snapshot_digest: Option<String>,
    pub launch_plan_digest: Option<String>,
    pub launch_nonce_blake3_hex: Option<String>,
    pub launch_challenge_nonce_blake3_hex: Option<String>,
    pub checkpoint_process_instance_id: String,
    pub checkpoint_monotonic_counter: u64,
    pub checkpoint_challenge_digest: Option<String>,
    pub fresh_observation_qualification_digest: Option<String>,
    pub mapped_object_set_digest: Option<String>,
    pub observed_at_ms: u64,
    pub disposition: LaunchBoundFirstObservationDisposition,
    pub issues: Vec<LaunchBoundFirstObservationIssue>,
}

impl LaunchBoundFirstObservationReport {
    pub fn canonical_digest(&self) -> String {
        let mut h = blake3::Hasher::new();
        h.update(REPORT_DOMAIN);
        for value in [
            self.schema_version.as_str(),
            self.policy_id.as_str(),
            self.policy_digest.as_deref().unwrap_or("-"),
            self.pipe_target.as_deref().unwrap_or("-"),
            self.pre_read_fd_observation_digest.as_deref().unwrap_or("-"),
            self.post_read_fd_observation_digest.as_deref().unwrap_or("-"),
            self.ticket_digest.as_deref().unwrap_or("-"),
            self.ticket_bytes_digest.as_deref().unwrap_or("-"),
            self.exec_confirmation_digest.as_deref().unwrap_or("-"),
            self.critical_authority_qualification_digest.as_deref().unwrap_or("-"),
            self.authority_snapshot_digest.as_deref().unwrap_or("-"),
            self.launch_plan_digest.as_deref().unwrap_or("-"),
            self.launch_nonce_blake3_hex.as_deref().unwrap_or("-"),
            self.launch_challenge_nonce_blake3_hex.as_deref().unwrap_or("-"),
            self.checkpoint_process_instance_id.as_str(),
            self.checkpoint_challenge_digest.as_deref().unwrap_or("-"),
            self.fresh_observation_qualification_digest.as_deref().unwrap_or("-"),
            self.mapped_object_set_digest.as_deref().unwrap_or("-"),
        ] {
            field(&mut h, value);
        }
        h.update(&self.current_pid.to_le_bytes());
        h.update(&self.handoff_read_fd.to_le_bytes());
        h.update(&self.ticket_bytes_len.to_le_bytes());
        h.update(&self.checkpoint_monotonic_counter.to_le_bytes());
        h.update(&self.observed_at_ms.to_le_bytes());
        field(
            &mut h,
            match self.disposition {
                LaunchBoundFirstObservationDisposition::Invalid => "invalid",
                LaunchBoundFirstObservationDisposition::Blocked => "blocked",
                LaunchBoundFirstObservationDisposition::Qualified => "qualified",
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
pub struct LaunchBoundFirstRuntimeObservation {
    qualification_digest: String,
    report_digest: String,
    policy_digest: String,
    current_pid: i32,
    handoff_read_fd: u32,
    pipe_target: String,
    ticket_digest: String,
    ticket_bytes_digest: String,
    exec_confirmation_digest: String,
    critical_authority_qualification_digest: String,
    authority_snapshot_digest: String,
    launch_plan_digest: String,
    launch_nonce_blake3_hex: String,
    launch_challenge_nonce_blake3_hex: String,
    process_instance_id: String,
    checkpoint_monotonic_counter: u64,
    checkpoint_challenge_digest: String,
    fresh_observation_qualification_digest: String,
    raw_observation_qualification_digest: String,
    mapped_object_set_digest: String,
    runtime_policy_digest: String,
    runtime_verifier_ref: String,
    backend_id: String,
    executable_digest: String,
    observed_at_ms: u64,
}

impl LaunchBoundFirstRuntimeObservation {
    pub fn qualification_digest(&self) -> &str { &self.qualification_digest }
    pub fn report_digest(&self) -> &str { &self.report_digest }
    pub fn policy_digest(&self) -> &str { &self.policy_digest }
    pub const fn current_pid(&self) -> i32 { self.current_pid }
    pub const fn handoff_read_fd(&self) -> u32 { self.handoff_read_fd }
    pub fn pipe_target(&self) -> &str { &self.pipe_target }
    pub fn ticket_digest(&self) -> &str { &self.ticket_digest }
    pub fn ticket_bytes_digest(&self) -> &str { &self.ticket_bytes_digest }
    pub fn exec_confirmation_digest(&self) -> &str { &self.exec_confirmation_digest }
    pub fn critical_authority_qualification_digest(&self) -> &str {
        &self.critical_authority_qualification_digest
    }
    pub fn authority_snapshot_digest(&self) -> &str { &self.authority_snapshot_digest }
    pub fn launch_plan_digest(&self) -> &str { &self.launch_plan_digest }
    pub fn launch_nonce_blake3_hex(&self) -> &str { &self.launch_nonce_blake3_hex }
    pub fn launch_challenge_nonce_blake3_hex(&self) -> &str {
        &self.launch_challenge_nonce_blake3_hex
    }
    pub fn process_instance_id(&self) -> &str { &self.process_instance_id }
    pub const fn checkpoint_monotonic_counter(&self) -> u64 { self.checkpoint_monotonic_counter }
    pub fn checkpoint_challenge_digest(&self) -> &str { &self.checkpoint_challenge_digest }
    pub fn fresh_observation_qualification_digest(&self) -> &str {
        &self.fresh_observation_qualification_digest
    }
    pub fn raw_observation_qualification_digest(&self) -> &str {
        &self.raw_observation_qualification_digest
    }
    pub fn mapped_object_set_digest(&self) -> &str { &self.mapped_object_set_digest }
    pub fn runtime_policy_digest(&self) -> &str { &self.runtime_policy_digest }
    pub fn runtime_verifier_ref(&self) -> &str { &self.runtime_verifier_ref }
    pub fn backend_id(&self) -> &str { &self.backend_id }
    pub fn executable_digest(&self) -> &str { &self.executable_digest }
    pub const fn observed_at_ms(&self) -> u64 { self.observed_at_ms }

    pub const fn canonical_ticket_consumed_from_reviewed_inherited_fd(&self) -> bool { true }
    pub const fn consuming_os_pid_matches_ticket_tracee_pid(&self) -> bool { true }
    pub const fn checkpoint_one_challenge_derived_from_consumed_ticket(&self) -> bool { true }
    pub const fn checkpoint_one_has_no_predecessor(&self) -> bool { true }
    pub const fn live_mapped_observation_performed_after_ticket_consumption(&self) -> bool { true }
    pub const fn supervisor_issued_ticket_authenticated(&self) -> bool { false }
    pub const fn exclusive_pipe_writer_authority_established(&self) -> bool { false }
    pub const fn first_signed_checkpoint_binding_established(&self) -> bool { false }
    pub const fn uninterrupted_mapping_continuity_since_exec_established(&self) -> bool { false }
    pub const fn ticket_consumption_was_first_application_action_established(&self) -> bool { false }
    pub const fn trusted_time_established(&self) -> bool { false }
    pub const fn grants_physical_authority(&self) -> bool { false }
}

pub struct LaunchBoundFirstObservationQualification {
    pub report: LaunchBoundFirstObservationReport,
    pub ticket: LaunchRuntimeHandoffTicket,
    pub fresh: FreshMappedRuntimeObservationQualification,
    verified: LaunchBoundFirstRuntimeObservation,
}

impl LaunchBoundFirstObservationQualification {
    pub fn verified(&self) -> &LaunchBoundFirstRuntimeObservation { &self.verified }
    pub fn into_verified(self) -> LaunchBoundFirstRuntimeObservation { self.verified }
}

#[allow(clippy::too_many_arguments)]
pub fn consume_launch_ticket_and_observe_checkpoint_one(
    policy: &LaunchBoundFirstObservationPolicy,
    fresh_policy: &FreshMappedRuntimePolicy,
    mapped_policy: &MappedExecutableRuntimePolicy,
    parent: &ClosureBackedInProcessContinuousVerifierExecution,
    bound: &NixBoundInProcessRuntimePolicy,
    closure: &NixRuntimeClosureQualification,
    checkpoint_monotonic_counter: u64,
    observed_at_ms: u64,
) -> Result<LaunchBoundFirstObservationQualification, LaunchBoundFirstObservationReport> {
    let policy_digest = policy.canonical_digest();
    let current_pid = i32::try_from(std::process::id()).unwrap_or(-1);
    let mut report = LaunchBoundFirstObservationReport {
        schema_version: LAUNCH_BOUND_FIRST_OBSERVATION_REPORT_SCHEMA_V1.into(),
        policy_id: policy.policy_id.clone(),
        policy_digest: policy_digest.clone(),
        current_pid,
        handoff_read_fd: policy.handoff_read_fd,
        pipe_target: None,
        pre_read_fd_observation_digest: None,
        post_read_fd_observation_digest: None,
        ticket_digest: None,
        ticket_bytes_digest: None,
        ticket_bytes_len: 0,
        exec_confirmation_digest: None,
        critical_authority_qualification_digest: None,
        authority_snapshot_digest: None,
        launch_plan_digest: None,
        launch_nonce_blake3_hex: None,
        launch_challenge_nonce_blake3_hex: None,
        checkpoint_process_instance_id: parent.process_instance_id().into(),
        checkpoint_monotonic_counter,
        checkpoint_challenge_digest: None,
        fresh_observation_qualification_digest: None,
        mapped_object_set_digest: None,
        observed_at_ms,
        disposition: LaunchBoundFirstObservationDisposition::Invalid,
        issues: Vec::new(),
    };

    if !policy.validate() {
        report.issues.push(LaunchBoundFirstObservationIssue::InvalidPolicy);
        return Err(finalize(report));
    }
    if checkpoint_monotonic_counter == 0 {
        report
            .issues
            .push(LaunchBoundFirstObservationIssue::InvalidCheckpointCounter);
    }
    if fresh_policy.canonical_digest().as_deref()
        != Some(policy.expected_fresh_mapped_policy_digest.as_str())
    {
        report
            .issues
            .push(LaunchBoundFirstObservationIssue::FreshMappedPolicyMismatch);
    }
    if parent.policy_digest() != policy.expected_closure_backed_policy_digest {
        report
            .issues
            .push(LaunchBoundFirstObservationIssue::ClosureBackedPolicyMismatch);
    }
    if bound.policy_digest() != policy.expected_nix_binding_policy_digest {
        report
            .issues
            .push(LaunchBoundFirstObservationIssue::NixBindingPolicyMismatch);
    }
    if mapped_policy.canonical_digest().as_deref()
        != Some(policy.expected_mapped_runtime_policy_digest.as_str())
    {
        report
            .issues
            .push(LaunchBoundFirstObservationIssue::MappedRuntimePolicyMismatch);
    }
    if parent.runtime_policy_digest() != policy.expected_runtime_policy_digest
        || bound.runtime_policy_digest() != policy.expected_runtime_policy_digest
    {
        report.issues.push(LaunchBoundFirstObservationIssue::RuntimePolicyMismatch);
    }
    if parent.runtime_verifier_ref() != policy.expected_runtime_verifier_ref
        || bound.runtime_verifier_ref() != policy.expected_runtime_verifier_ref
    {
        report
            .issues
            .push(LaunchBoundFirstObservationIssue::RuntimeVerifierMismatch);
    }
    if parent.backend_id() != policy.expected_backend_id || bound.backend_id() != policy.expected_backend_id {
        report.issues.push(LaunchBoundFirstObservationIssue::BackendMismatch);
    }
    let fd = match i32::try_from(policy.handoff_read_fd) {
        Ok(value) if value >= 0 => value,
        _ => {
            report.issues.push(LaunchBoundFirstObservationIssue::HandoffFdInvalid);
            return Err(finalize(report));
        }
    };
    if !report.issues.is_empty() {
        return Err(finalize(report));
    }

    let before = match observe_pipe_fd(fd, policy.max_fdinfo_bytes) {
        Ok(value) => value,
        Err(error) => {
            report
                .issues
                .push(LaunchBoundFirstObservationIssue::HandoffFdUnavailable(error));
            return Err(finalize(report));
        }
    };
    report.pipe_target = Some(before.target.clone());
    report.pre_read_fd_observation_digest = Some(before.observation_digest.clone());
    if parse_pipe_target(&before.target).is_none() {
        report
            .issues
            .push(LaunchBoundFirstObservationIssue::HandoffFdTargetNotPipe);
    }
    if access_mode(before.flags) != OFlag::O_RDONLY.bits() {
        report
            .issues
            .push(LaunchBoundFirstObservationIssue::HandoffFdNotReadOnly);
    }
    if before.flags & OFlag::O_CLOEXEC.bits() != 0 {
        report
            .issues
            .push(LaunchBoundFirstObservationIssue::HandoffFdCloseOnExecUnexpected);
    }
    if before.flags & OFlag::O_NONBLOCK.bits() != 0 {
        report
            .issues
            .push(LaunchBoundFirstObservationIssue::HandoffFdNonBlocking);
    }
    if parse_pipe_target(&before.target) != Some(before.inode) {
        report
            .issues
            .push(LaunchBoundFirstObservationIssue::HandoffFdInfoInconsistent);
    }
    if !report.issues.is_empty() {
        return Err(finalize(report));
    }

    let wire = match read_one_ticket(fd, policy.max_ticket_bytes) {
        Ok(value) => value,
        Err(error) => {
            report
                .issues
                .push(LaunchBoundFirstObservationIssue::TicketReadFailed(error));
            return Err(finalize(report));
        }
    };
    report.ticket_bytes_len = wire.len() as u64;
    report.ticket_bytes_digest = Some(format!("blake3:{}", blake3::hash(&wire).to_hex()));
    if wire.len() as u32 > policy.max_ticket_bytes {
        report.issues.push(LaunchBoundFirstObservationIssue::TicketTooLarge {
            observed: wire.len() as u64,
            maximum: policy.max_ticket_bytes as u64,
        });
        return Err(finalize(report));
    }
    let ticket = match LaunchRuntimeHandoffTicket::from_wire_bytes(&wire) {
        Some(value) => value,
        None => {
            report.issues.push(LaunchBoundFirstObservationIssue::TicketDecodeFailed);
            return Err(finalize(report));
        }
    };
    report.ticket_digest = Some(ticket.ticket_digest.clone());
    report.exec_confirmation_digest = Some(ticket.exec_confirmation_digest.clone());
    report.critical_authority_qualification_digest =
        Some(ticket.critical_authority_qualification_digest.clone());
    report.authority_snapshot_digest = Some(ticket.authority_snapshot_digest.clone());
    report.launch_plan_digest = Some(ticket.plan_digest.clone());
    report.launch_nonce_blake3_hex = Some(ticket.launch_nonce_blake3_hex.clone());
    report.launch_challenge_nonce_blake3_hex = Some(ticket.challenge_nonce_blake3_hex.clone());

    if ticket.tracee_handoff_read_fd != policy.handoff_read_fd {
        report
            .issues
            .push(LaunchBoundFirstObservationIssue::TicketReadFdMismatch);
    }
    if ticket.pipe_target != before.target {
        report
            .issues
            .push(LaunchBoundFirstObservationIssue::TicketPipeTargetMismatch);
    }
    if ticket.tracee_pid != current_pid {
        report.issues.push(LaunchBoundFirstObservationIssue::TicketPidMismatch {
            expected: ticket.tracee_pid,
            observed: current_pid,
        });
    }
    if ticket.runtime_policy_digest != parent.runtime_policy_digest() {
        report
            .issues
            .push(LaunchBoundFirstObservationIssue::TicketRuntimePolicyMismatch);
    }
    if ticket.runtime_verifier_ref != parent.runtime_verifier_ref() {
        report
            .issues
            .push(LaunchBoundFirstObservationIssue::TicketRuntimeVerifierMismatch);
    }
    if ticket.backend_id != parent.backend_id() {
        report
            .issues
            .push(LaunchBoundFirstObservationIssue::TicketBackendMismatch);
    }
    if ticket.executable_digest != parent.executable_digest() {
        report
            .issues
            .push(LaunchBoundFirstObservationIssue::TicketExecutableDigestMismatch);
    }

    let after = match observe_pipe_fd(fd, policy.max_fdinfo_bytes) {
        Ok(value) => value,
        Err(error) => {
            report
                .issues
                .push(LaunchBoundFirstObservationIssue::HandoffFdUnavailable(error));
            return Err(finalize(report));
        }
    };
    report.post_read_fd_observation_digest = Some(after.observation_digest.clone());
    if before != after {
        report
            .issues
            .push(LaunchBoundFirstObservationIssue::HandoffFdChangedDuringConsumption);
    }
    if !report.issues.is_empty() {
        return Err(finalize(report));
    }

    let challenge = MappedRuntimeCheckpointChallenge {
        schema_version:
            symthaea_assurance_fresh_mapped_runtime_continuity::MAPPED_RUNTIME_CHECKPOINT_CHALLENGE_SCHEMA_V1
                .into(),
        process_instance_id: parent.process_instance_id().into(),
        runtime_policy_digest: parent.runtime_policy_digest().into(),
        verifier_ref: parent.runtime_verifier_ref().into(),
        checkpoint_sequence: 1,
        checkpoint_monotonic_counter,
        previous_checkpoint_digest: None,
        nonce_blake3_hex: ticket.challenge_nonce_blake3_hex.clone(),
    };
    let Some(challenge_digest) = challenge.canonical_digest() else {
        report
            .issues
            .push(LaunchBoundFirstObservationIssue::ChallengeConstructionFailed);
        return Err(finalize(report));
    };
    report.checkpoint_challenge_digest = Some(challenge_digest.clone());

    let fresh = match observe_fresh_mapped_runtime_for_checkpoint(
        fresh_policy,
        mapped_policy,
        &challenge,
        parent,
        bound,
        closure,
        observed_at_ms,
    ) {
        Ok(value) => value,
        Err(fresh_report) => {
            report
                .issues
                .push(LaunchBoundFirstObservationIssue::FreshObservationFailed(
                    fresh_report.canonical_digest(),
                ));
            return Err(finalize(report));
        }
    };
    let observed = fresh.fresh();
    report.fresh_observation_qualification_digest = Some(observed.qualification_digest().into());
    report.mapped_object_set_digest = Some(observed.mapped_object_set_digest().into());
    if observed.challenge_nonce_blake3_hex() != ticket.challenge_nonce_blake3_hex {
        report
            .issues
            .push(LaunchBoundFirstObservationIssue::FreshChallengeMismatch);
    }
    if observed.checkpoint_sequence() != 1 {
        report
            .issues
            .push(LaunchBoundFirstObservationIssue::FreshSequenceMismatch);
    }
    if observed.previous_checkpoint_digest().is_some() {
        report
            .issues
            .push(LaunchBoundFirstObservationIssue::FreshPredecessorMismatch);
    }
    if observed.process_instance_id() != parent.process_instance_id() {
        report
            .issues
            .push(LaunchBoundFirstObservationIssue::FreshProcessMismatch);
    }
    if observed.runtime_policy_digest() != ticket.runtime_policy_digest {
        report
            .issues
            .push(LaunchBoundFirstObservationIssue::FreshRuntimePolicyMismatch);
    }
    if observed.runtime_verifier_ref() != ticket.runtime_verifier_ref {
        report
            .issues
            .push(LaunchBoundFirstObservationIssue::FreshVerifierMismatch);
    }
    if observed.backend_id() != ticket.backend_id {
        report
            .issues
            .push(LaunchBoundFirstObservationIssue::FreshBackendMismatch);
    }
    if observed.executable_digest() != ticket.executable_digest {
        report
            .issues
            .push(LaunchBoundFirstObservationIssue::FreshExecutableMismatch);
    }
    if !report.issues.is_empty() {
        return Err(finalize(report));
    }

    report.disposition = LaunchBoundFirstObservationDisposition::Qualified;
    let report_digest = report.canonical_digest();
    let policy_digest = policy_digest.expect("validated policy has digest");
    let qualification_digest = qualification_digest(
        &policy_digest,
        &ticket.ticket_digest,
        current_pid,
        &challenge_digest,
        observed.qualification_digest(),
        &report_digest,
    );
    let verified = LaunchBoundFirstRuntimeObservation {
        qualification_digest,
        report_digest,
        policy_digest,
        current_pid,
        handoff_read_fd: policy.handoff_read_fd,
        pipe_target: ticket.pipe_target.clone(),
        ticket_digest: ticket.ticket_digest.clone(),
        ticket_bytes_digest: report.ticket_bytes_digest.clone().expect("qualified report has ticket bytes"),
        exec_confirmation_digest: ticket.exec_confirmation_digest.clone(),
        critical_authority_qualification_digest: ticket
            .critical_authority_qualification_digest
            .clone(),
        authority_snapshot_digest: ticket.authority_snapshot_digest.clone(),
        launch_plan_digest: ticket.plan_digest.clone(),
        launch_nonce_blake3_hex: ticket.launch_nonce_blake3_hex.clone(),
        launch_challenge_nonce_blake3_hex: ticket.challenge_nonce_blake3_hex.clone(),
        process_instance_id: parent.process_instance_id().into(),
        checkpoint_monotonic_counter,
        checkpoint_challenge_digest: challenge_digest,
        fresh_observation_qualification_digest: observed.qualification_digest().into(),
        raw_observation_qualification_digest: observed
            .raw_observation_qualification_digest()
            .into(),
        mapped_object_set_digest: observed.mapped_object_set_digest().into(),
        runtime_policy_digest: parent.runtime_policy_digest().into(),
        runtime_verifier_ref: parent.runtime_verifier_ref().into(),
        backend_id: parent.backend_id().into(),
        executable_digest: parent.executable_digest().into(),
        observed_at_ms: observed.observed_at_ms(),
    };

    Ok(LaunchBoundFirstObservationQualification {
        report,
        ticket,
        fresh,
        verified,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct PipeFdObservation {
    target: String,
    flags: i32,
    inode: u64,
    raw_fdinfo_digest: String,
    observation_digest: String,
}

fn observe_pipe_fd(fd: RawFd, maximum: u64) -> Result<PipeFdObservation, String> {
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
    let raw_fdinfo_digest = format!("blake3:{}", blake3::hash(&bytes).to_hex());
    let parsed = parse_fdinfo(&bytes)?;
    let observation_digest = pipe_observation_digest(fd, &target, parsed.flags, parsed.inode, &raw_fdinfo_digest);
    Ok(PipeFdObservation {
        target,
        flags: parsed.flags,
        inode: parsed.inode,
        raw_fdinfo_digest,
        observation_digest,
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
                flags = Some(i32::from_str_radix(value, 8).map_err(|_| "invalid-flags")?);
            }
            "ino" => {
                if inode.is_some() {
                    return Err("duplicate-ino".into());
                }
                inode = Some(value.parse::<u64>().map_err(|_| "invalid-ino")?);
            }
            _ => {}
        }
    }
    Ok(ParsedFdInfo {
        flags: flags.ok_or("missing-flags")?,
        inode: inode.ok_or("missing-ino")?,
    })
}

fn read_one_ticket(fd: RawFd, maximum: u32) -> Result<Vec<u8>, String> {
    let mut out = Vec::with_capacity(maximum.min(1024) as usize);
    read_exact_append(fd, &mut out, WIRE_MAGIC.len(), maximum)?;
    if out.as_slice() != WIRE_MAGIC {
        return Err("wire-magic-mismatch".into());
    }
    for _ in 0..STRING_FIELDS_BEFORE_IDS {
        read_wire_field(fd, &mut out, maximum)?;
    }
    read_exact_append(fd, &mut out, 4, maximum)?;
    read_exact_append(fd, &mut out, 4, maximum)?;
    for _ in 0..STRING_FIELDS_AFTER_IDS {
        read_wire_field(fd, &mut out, maximum)?;
    }
    Ok(out)
}

fn read_wire_field(fd: RawFd, out: &mut Vec<u8>, maximum: u32) -> Result<(), String> {
    let prefix_start = out.len();
    read_exact_append(fd, out, 4, maximum)?;
    let prefix: [u8; 4] = out[prefix_start..prefix_start + 4]
        .try_into()
        .map_err(|_| "invalid-length-prefix")?;
    let length = u32::from_le_bytes(prefix);
    if length > maximum {
        return Err("wire-field-too-large".into());
    }
    read_exact_append(fd, out, length as usize, maximum)
}

fn read_exact_append(fd: RawFd, out: &mut Vec<u8>, count: usize, maximum: u32) -> Result<(), String> {
    let final_len = out
        .len()
        .checked_add(count)
        .ok_or_else(|| "wire-length-overflow".to_string())?;
    if final_len > maximum as usize {
        return Err(format!("wire-exceeds-limit:{final_len}:{maximum}"));
    }
    let start = out.len();
    out.resize(final_len, 0);
    let mut offset = start;
    while offset < final_len {
        match read(fd, &mut out[offset..final_len]) {
            Ok(0) => return Err("unexpected-eof".into()),
            Ok(n) => offset += n,
            Err(error) => return Err(error.to_string()),
        }
    }
    Ok(())
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

fn pipe_observation_digest(
    fd: RawFd,
    target: &str,
    flags: i32,
    inode: u64,
    fdinfo_digest: &str,
) -> String {
    let mut h = blake3::Hasher::new();
    h.update(PIPE_OBSERVATION_DOMAIN);
    h.update(&fd.to_le_bytes());
    field(&mut h, target);
    h.update(&flags.to_le_bytes());
    h.update(&inode.to_le_bytes());
    field(&mut h, fdinfo_digest);
    format!("blake3:{}", h.finalize().to_hex())
}

fn qualification_digest(
    policy_digest: &str,
    ticket_digest: &str,
    current_pid: i32,
    challenge_digest: &str,
    fresh_observation_digest: &str,
    report_digest: &str,
) -> String {
    let mut h = blake3::Hasher::new();
    h.update(QUALIFICATION_DOMAIN);
    for value in [
        policy_digest,
        ticket_digest,
        challenge_digest,
        fresh_observation_digest,
        report_digest,
    ] {
        field(&mut h, value);
    }
    h.update(&current_pid.to_le_bytes());
    format!("blake3:{}", h.finalize().to_hex())
}

fn finalize(mut report: LaunchBoundFirstObservationReport) -> LaunchBoundFirstObservationReport {
    report.disposition = if report.issues.iter().any(LaunchBoundFirstObservationIssue::invalid) {
        LaunchBoundFirstObservationDisposition::Invalid
    } else {
        LaunchBoundFirstObservationDisposition::Blocked
    };
    report
}

fn parse_pipe_target(value: &str) -> Option<u64> {
    let body = value.strip_prefix("pipe:[")?.strip_suffix(']')?;
    if body.is_empty() || !body.bytes().all(|byte| byte.is_ascii_digit()) {
        return None;
    }
    body.parse::<u64>().ok()
}

fn access_mode(flags: i32) -> i32 { flags & OFlag::O_ACCMODE.bits() }

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

    fn policy() -> LaunchBoundFirstObservationPolicy {
        LaunchBoundFirstObservationPolicy {
            schema_version: LAUNCH_BOUND_FIRST_OBSERVATION_POLICY_SCHEMA_V1.into(),
            policy_id: "policy:launch-bound-first-observation:1".into(),
            expected_fresh_mapped_policy_digest: d("fresh-policy"),
            expected_closure_backed_policy_digest: d("closure-backed-policy"),
            expected_nix_binding_policy_digest: d("nix-binding-policy"),
            expected_mapped_runtime_policy_digest: d("mapped-policy"),
            expected_runtime_policy_digest: d("runtime-policy"),
            expected_runtime_verifier_ref: "verifier:host".into(),
            expected_backend_id: "backend:in-process".into(),
            handoff_read_fd: 7,
            max_ticket_bytes: 4096,
            max_fdinfo_bytes: 1 << 20,
            evidence_refs: vec!["review:a".into(), "review:b".into()],
        }
    }

    #[test]
    fn policy_reference_order_is_nonsemantic() {
        let left = policy();
        let mut right = left.clone();
        right.evidence_refs.reverse();
        assert_eq!(left.canonical_digest(), right.canonical_digest());
        right.handoff_read_fd += 1;
        assert_ne!(left.canonical_digest(), right.canonical_digest());
    }

    #[test]
    fn pipe_target_parser_is_strict() {
        assert_eq!(parse_pipe_target("pipe:[123]"), Some(123));
        assert_eq!(parse_pipe_target("socket:[123]"), None);
        assert_eq!(parse_pipe_target("pipe:[]"), None);
        assert_eq!(parse_pipe_target("pipe:[12x]"), None);
    }

    #[test]
    fn qualification_identity_binds_pid_ticket_challenge_and_observation() {
        let a = qualification_digest(&d("policy"), &d("ticket"), 42, &d("challenge"), &d("fresh"), &d("report"));
        let b = qualification_digest(&d("policy"), &d("ticket"), 43, &d("challenge"), &d("fresh"), &d("report"));
        assert_ne!(a, b);
    }

    #[test]
    fn claim_ceiling_remains_explicit() {
        let claims = [
            "supervisor_issued_ticket_authenticated=false",
            "exclusive_pipe_writer_authority_established=false",
            "first_signed_checkpoint_binding_established=false",
            "uninterrupted_mapping_continuity_since_exec_established=false",
            "ticket_consumption_was_first_application_action_established=false",
            "trusted_time_established=false",
            "grants_physical_authority=false",
        ];
        assert_eq!(claims.len(), 7);
    }

    #[test]
    fn wire_shape_constants_match_parent_ticket_format() {
        assert_eq!(WIRE_MAGIC, b"SYMT-LR-HANDOFF-V1\0");
        assert_eq!(STRING_FIELDS_BEFORE_IDS, 6);
        assert_eq!(STRING_FIELDS_AFTER_IDS, 8);
    }
}
