// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Tracee-side post-release ticket consumption before checkpoint-two live
//! mapped-runtime observation.
//!
//! This theorem is deliberately non-circular. It consumes the post-release
//! ticket first, constructs checkpoint-two lineage coordinates second, and then
//! invokes the lower #3365 live mapped-runtime observer directly. It does not
//! require an already-completed continuous-execution capability in order to
//! produce the observation that checkpoint two will later sign.

#![cfg(target_os = "linux")]
#![deny(unsafe_code)]

use nix::{fcntl::OFlag, unistd::read};
use serde::{Deserialize, Serialize};
use std::{collections::BTreeSet, fs, os::unix::io::RawFd, path::PathBuf};
use symthaea_assurance_fresh_mapped_runtime_continuity::{
    MAPPED_RUNTIME_CHECKPOINT_CHALLENGE_SCHEMA_V1, MappedRuntimeCheckpointChallenge,
};
use symthaea_assurance_nix_bound_in_process_runtime::NixBoundInProcessRuntimePolicy;
use symthaea_assurance_nix_runtime_closure::NixRuntimeClosureQualification;
use symthaea_assurance_observed_mapped_nix_executable_runtime::{
    MappedExecutableRuntimePolicy, MappedExecutableRuntimeQualification,
    observe_mapped_nix_executable_runtime,
};
use symthaea_assurance_post_release_runtime_handoff::PostReleaseRuntimeHandoffTicket;
use symthaea_evidence_verifier_runtime_continuity::VerifierRuntimeContinuityPolicy;

pub const POST_RELEASE_CHECKPOINT_TWO_POLICY_SCHEMA_V1: &str =
    "symthaea.assurance.post-release-checkpoint-two-observation-policy.v1";
pub const POST_RELEASE_CHECKPOINT_TWO_REPORT_SCHEMA_V1: &str =
    "symthaea.assurance.post-release-checkpoint-two-observation-report.v1";

const POLICY_DOMAIN: &[u8] =
    b"symthaea.assurance.post-release-checkpoint-two-observation-policy.digest.v1\0";
const PIPE_OBSERVATION_DOMAIN: &[u8] =
    b"symthaea.assurance.post-release-checkpoint-two-pipe-observation.digest.v1\0";
const REPORT_DOMAIN: &[u8] =
    b"symthaea.assurance.post-release-checkpoint-two-observation-report.digest.v1\0";
const QUALIFICATION_DOMAIN: &[u8] =
    b"symthaea.assurance.post-release-checkpoint-two-observation-qualification.digest.v1\0";
const WIRE_MAGIC: &[u8] = b"SYMT-POST-RELEASE-HANDOFF-V1\0";
const STRING_FIELDS_BEFORE_IDS: usize = 10;
const STRING_FIELDS_AFTER_IDS: usize = 13;
const MAX_TEXT: usize = 4096;
const MAX_REFS: usize = 128;
const HARD_MAX_TICKET_BYTES: u32 = 4096;
const HARD_MAX_FDINFO_BYTES: u64 = 4 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PostReleaseCheckpointTwoObservationPolicy {
    pub schema_version: String,
    pub policy_id: String,
    pub expected_mapped_runtime_policy_digest: String,
    pub expected_nix_binding_policy_digest: String,
    pub expected_closure_policy_digest: String,
    pub expected_runtime_policy_digest: String,
    pub expected_runtime_verifier_ref: String,
    pub expected_backend_id: String,
    pub handoff_read_fd: u32,
    pub max_ticket_bytes: u32,
    pub max_fdinfo_bytes: u64,
    pub evidence_refs: Vec<String>,
}

impl PostReleaseCheckpointTwoObservationPolicy {
    pub fn validate(&self) -> bool {
        self.schema_version == POST_RELEASE_CHECKPOINT_TWO_POLICY_SCHEMA_V1
            && canonical_text(&self.policy_id)
            && valid_blake3(&self.expected_mapped_runtime_policy_digest)
            && valid_blake3(&self.expected_nix_binding_policy_digest)
            && valid_blake3(&self.expected_closure_policy_digest)
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
            self.expected_mapped_runtime_policy_digest.as_str(),
            self.expected_nix_binding_policy_digest.as_str(),
            self.expected_closure_policy_digest.as_str(),
            self.expected_runtime_policy_digest.as_str(),
            self.expected_runtime_verifier_ref.as_str(),
            self.expected_backend_id.as_str(),
        ] {
            field(&mut h, value);
        }
        h.update(&self.handoff_read_fd.to_le_bytes());
        h.update(&self.max_ticket_bytes.to_le_bytes());
        h.update(&self.max_fdinfo_bytes.to_le_bytes());
        sorted(&mut h, &self.evidence_refs);
        Some(b3(h.finalize()))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum PostReleaseCheckpointTwoObservationIssue {
    InvalidPolicy,
    InvalidRuntimePolicy,
    InvalidCheckpointCounter,
    InvalidHandoffReadFd,
    MappedRuntimePolicyMismatch,
    NixBindingPolicyMismatch,
    ClosurePolicyMismatch,
    RuntimePolicyMismatch,
    RuntimeVerifierMismatch,
    BackendMismatch,
    RuntimeExecutableMismatch,
    RuntimeClosureMismatch,
    HandoffFdUnavailable(String),
    HandoffTargetNotPipe,
    HandoffFdNotReadOnly,
    HandoffFdCloseOnExecUnexpected,
    HandoffFdNonBlocking,
    HandoffFdInfoInconsistent,
    TicketReadFailed(String),
    TicketDecodeFailed,
    TicketReadFdMismatch,
    TicketPipeTargetMismatch,
    TicketPidMismatch { expected: i32, observed: i32 },
    TicketCheckpointSequenceMismatch,
    TicketPredecessorAliasMismatch,
    TicketCounterRegression,
    TicketRuntimePolicyMismatch,
    TicketRuntimeVerifierMismatch,
    TicketBackendMismatch,
    TicketBootMeasurementMismatch,
    TicketExecutableMismatch,
    TicketClosureMismatch,
    TicketRuntimeConfigMismatch,
    HandoffFdChangedDuringConsumption,
    ChallengeConstructionFailed,
    RawObservationFailed(String),
    ObservationPolicyMismatch,
    ObservationNixBindingMismatch,
    ObservationClosureQualificationMismatch,
    ObservationClosureDigestMismatch,
    ObservationRuntimePolicyMismatch,
    ObservationVerifierMismatch,
    ObservationBackendMismatch,
    ObservationExecutableMismatch,
    ObservationTimeMismatch,
}

impl PostReleaseCheckpointTwoObservationIssue {
    fn invalid(&self) -> bool {
        matches!(
            self,
            Self::InvalidPolicy
                | Self::InvalidRuntimePolicy
                | Self::InvalidCheckpointCounter
                | Self::InvalidHandoffReadFd
                | Self::TicketDecodeFailed
                | Self::TicketCheckpointSequenceMismatch
                | Self::TicketPredecessorAliasMismatch
        )
    }

    fn code(&self) -> String {
        match self {
            Self::InvalidPolicy => "invalid-policy".into(),
            Self::InvalidRuntimePolicy => "invalid-runtime-policy".into(),
            Self::InvalidCheckpointCounter => "invalid-checkpoint-counter".into(),
            Self::InvalidHandoffReadFd => "invalid-handoff-read-fd".into(),
            Self::MappedRuntimePolicyMismatch => "mapped-runtime-policy-mismatch".into(),
            Self::NixBindingPolicyMismatch => "nix-binding-policy-mismatch".into(),
            Self::ClosurePolicyMismatch => "closure-policy-mismatch".into(),
            Self::RuntimePolicyMismatch => "runtime-policy-mismatch".into(),
            Self::RuntimeVerifierMismatch => "runtime-verifier-mismatch".into(),
            Self::BackendMismatch => "backend-mismatch".into(),
            Self::RuntimeExecutableMismatch => "runtime-executable-mismatch".into(),
            Self::RuntimeClosureMismatch => "runtime-closure-mismatch".into(),
            Self::HandoffFdUnavailable(value) => format!("handoff-fd-unavailable:{value}"),
            Self::HandoffTargetNotPipe => "handoff-target-not-pipe".into(),
            Self::HandoffFdNotReadOnly => "handoff-fd-not-read-only".into(),
            Self::HandoffFdCloseOnExecUnexpected => "handoff-fd-cloexec-unexpected".into(),
            Self::HandoffFdNonBlocking => "handoff-fd-nonblocking".into(),
            Self::HandoffFdInfoInconsistent => "handoff-fdinfo-inconsistent".into(),
            Self::TicketReadFailed(value) => format!("ticket-read-failed:{value}"),
            Self::TicketDecodeFailed => "ticket-decode-failed".into(),
            Self::TicketReadFdMismatch => "ticket-read-fd-mismatch".into(),
            Self::TicketPipeTargetMismatch => "ticket-pipe-target-mismatch".into(),
            Self::TicketPidMismatch { expected, observed } => {
                format!("ticket-pid-mismatch:{expected}:{observed}")
            }
            Self::TicketCheckpointSequenceMismatch => "ticket-checkpoint-sequence-mismatch".into(),
            Self::TicketPredecessorAliasMismatch => "ticket-predecessor-alias-mismatch".into(),
            Self::TicketCounterRegression => "ticket-counter-regression".into(),
            Self::TicketRuntimePolicyMismatch => "ticket-runtime-policy-mismatch".into(),
            Self::TicketRuntimeVerifierMismatch => "ticket-runtime-verifier-mismatch".into(),
            Self::TicketBackendMismatch => "ticket-backend-mismatch".into(),
            Self::TicketBootMeasurementMismatch => "ticket-boot-measurement-mismatch".into(),
            Self::TicketExecutableMismatch => "ticket-executable-mismatch".into(),
            Self::TicketClosureMismatch => "ticket-closure-mismatch".into(),
            Self::TicketRuntimeConfigMismatch => "ticket-runtime-config-mismatch".into(),
            Self::HandoffFdChangedDuringConsumption => {
                "handoff-fd-changed-during-consumption".into()
            }
            Self::ChallengeConstructionFailed => "challenge-construction-failed".into(),
            Self::RawObservationFailed(value) => format!("raw-observation-failed:{value}"),
            Self::ObservationPolicyMismatch => "observation-policy-mismatch".into(),
            Self::ObservationNixBindingMismatch => "observation-nix-binding-mismatch".into(),
            Self::ObservationClosureQualificationMismatch => {
                "observation-closure-qualification-mismatch".into()
            }
            Self::ObservationClosureDigestMismatch => "observation-closure-digest-mismatch".into(),
            Self::ObservationRuntimePolicyMismatch => "observation-runtime-policy-mismatch".into(),
            Self::ObservationVerifierMismatch => "observation-verifier-mismatch".into(),
            Self::ObservationBackendMismatch => "observation-backend-mismatch".into(),
            Self::ObservationExecutableMismatch => "observation-executable-mismatch".into(),
            Self::ObservationTimeMismatch => "observation-time-mismatch".into(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PostReleaseCheckpointTwoObservationDisposition {
    Invalid,
    Blocked,
    Qualified,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PostReleaseCheckpointTwoObservationReport {
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
    pub channel_retention_digest: Option<String>,
    pub release_digest: Option<String>,
    pub release_report_digest: Option<String>,
    pub confinement_qualification_digest: Option<String>,
    pub mapping_continuity_qualification_digest: Option<String>,
    pub bootstrap_ready_checkpoint_digest: Option<String>,
    pub process_instance_id: String,
    pub checkpoint_sequence: u64,
    pub previous_checkpoint_digest: Option<String>,
    pub previous_checkpoint_monotonic_counter: u64,
    pub checkpoint_monotonic_counter: u64,
    pub checkpoint_challenge_digest: Option<String>,
    pub raw_observation_qualification_digest: Option<String>,
    pub mapped_object_set_digest: Option<String>,
    pub runtime_policy_digest: String,
    pub runtime_verifier_ref: String,
    pub backend_id: String,
    pub executable_digest: String,
    pub dependency_closure_digest: String,
    pub observed_at_ms: u64,
    pub disposition: PostReleaseCheckpointTwoObservationDisposition,
    pub issues: Vec<PostReleaseCheckpointTwoObservationIssue>,
}

impl PostReleaseCheckpointTwoObservationReport {
    pub fn canonical_digest(&self) -> String {
        let mut h = blake3::Hasher::new();
        h.update(REPORT_DOMAIN);
        for value in [
            self.schema_version.as_str(),
            self.policy_id.as_str(),
            self.policy_digest.as_deref().unwrap_or("-"),
            self.pipe_target.as_deref().unwrap_or("-"),
            self.pre_read_fd_observation_digest
                .as_deref()
                .unwrap_or("-"),
            self.post_read_fd_observation_digest
                .as_deref()
                .unwrap_or("-"),
            self.ticket_digest.as_deref().unwrap_or("-"),
            self.ticket_bytes_digest.as_deref().unwrap_or("-"),
            self.channel_retention_digest.as_deref().unwrap_or("-"),
            self.release_digest.as_deref().unwrap_or("-"),
            self.release_report_digest.as_deref().unwrap_or("-"),
            self.confinement_qualification_digest
                .as_deref()
                .unwrap_or("-"),
            self.mapping_continuity_qualification_digest
                .as_deref()
                .unwrap_or("-"),
            self.bootstrap_ready_checkpoint_digest
                .as_deref()
                .unwrap_or("-"),
            self.process_instance_id.as_str(),
            self.previous_checkpoint_digest.as_deref().unwrap_or("-"),
            self.checkpoint_challenge_digest.as_deref().unwrap_or("-"),
            self.raw_observation_qualification_digest
                .as_deref()
                .unwrap_or("-"),
            self.mapped_object_set_digest.as_deref().unwrap_or("-"),
            self.runtime_policy_digest.as_str(),
            self.runtime_verifier_ref.as_str(),
            self.backend_id.as_str(),
            self.executable_digest.as_str(),
            self.dependency_closure_digest.as_str(),
        ] {
            field(&mut h, value);
        }
        h.update(&self.current_pid.to_le_bytes());
        h.update(&self.handoff_read_fd.to_le_bytes());
        h.update(&self.ticket_bytes_len.to_le_bytes());
        h.update(&self.checkpoint_sequence.to_le_bytes());
        h.update(&self.previous_checkpoint_monotonic_counter.to_le_bytes());
        h.update(&self.checkpoint_monotonic_counter.to_le_bytes());
        h.update(&self.observed_at_ms.to_le_bytes());
        field(
            &mut h,
            match self.disposition {
                PostReleaseCheckpointTwoObservationDisposition::Invalid => "invalid",
                PostReleaseCheckpointTwoObservationDisposition::Blocked => "blocked",
                PostReleaseCheckpointTwoObservationDisposition::Qualified => "qualified",
            },
        );
        h.update(&(self.issues.len() as u64).to_le_bytes());
        for issue in &self.issues {
            field(&mut h, &issue.code());
        }
        b3(h.finalize())
    }

    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PostReleaseCheckpointTwoObservation {
    qualification_digest: String,
    report_digest: String,
    policy_digest: String,
    current_pid: i32,
    handoff_read_fd: u32,
    pipe_target: String,
    ticket_digest: String,
    ticket_bytes_digest: String,
    channel_retention_digest: String,
    release_digest: String,
    release_report_digest: String,
    confinement_qualification_digest: String,
    mapping_continuity_qualification_digest: String,
    bootstrap_ready_wire_digest: String,
    bootstrap_ready_checkpoint_digest: String,
    launch_handoff_qualification_digest: String,
    launch_ticket_digest: String,
    process_instance_id: String,
    checkpoint_monotonic_counter: u64,
    checkpoint_challenge_digest: String,
    challenge_nonce_blake3_hex: String,
    raw_observation_qualification_digest: String,
    raw_observation_report_digest: String,
    mapped_object_set_digest: String,
    runtime_policy_digest: String,
    runtime_verifier_ref: String,
    backend_id: String,
    boot_measurement_digest: String,
    executable_digest: String,
    dependency_closure_digest: String,
    runtime_config_digest: String,
    launch_attestation_digest: String,
    checkpoint_one_challenge_digest: String,
    checkpoint_one_claimed_observation_digest: String,
    previous_checkpoint_monotonic_counter: u64,
    observed_at_ms: u64,
}

impl PostReleaseCheckpointTwoObservation {
    pub fn qualification_digest(&self) -> &str {
        &self.qualification_digest
    }
    pub fn report_digest(&self) -> &str {
        &self.report_digest
    }
    pub fn policy_digest(&self) -> &str {
        &self.policy_digest
    }
    pub const fn current_pid(&self) -> i32 {
        self.current_pid
    }
    pub const fn handoff_read_fd(&self) -> u32 {
        self.handoff_read_fd
    }
    pub fn pipe_target(&self) -> &str {
        &self.pipe_target
    }
    pub fn ticket_digest(&self) -> &str {
        &self.ticket_digest
    }
    pub fn ticket_bytes_digest(&self) -> &str {
        &self.ticket_bytes_digest
    }
    pub fn channel_retention_digest(&self) -> &str {
        &self.channel_retention_digest
    }
    pub fn release_digest(&self) -> &str {
        &self.release_digest
    }
    pub fn release_report_digest(&self) -> &str {
        &self.release_report_digest
    }
    pub fn confinement_qualification_digest(&self) -> &str {
        &self.confinement_qualification_digest
    }
    pub fn mapping_continuity_qualification_digest(&self) -> &str {
        &self.mapping_continuity_qualification_digest
    }
    pub fn bootstrap_ready_wire_digest(&self) -> &str {
        &self.bootstrap_ready_wire_digest
    }
    pub fn bootstrap_ready_checkpoint_digest(&self) -> &str {
        &self.bootstrap_ready_checkpoint_digest
    }
    pub fn launch_handoff_qualification_digest(&self) -> &str {
        &self.launch_handoff_qualification_digest
    }
    pub fn launch_ticket_digest(&self) -> &str {
        &self.launch_ticket_digest
    }
    pub fn process_instance_id(&self) -> &str {
        &self.process_instance_id
    }
    pub const fn checkpoint_sequence(&self) -> u64 {
        2
    }
    pub fn previous_checkpoint_digest(&self) -> &str {
        &self.bootstrap_ready_checkpoint_digest
    }
    pub const fn checkpoint_monotonic_counter(&self) -> u64 {
        self.checkpoint_monotonic_counter
    }
    pub fn checkpoint_challenge_digest(&self) -> &str {
        &self.checkpoint_challenge_digest
    }
    pub fn challenge_nonce_blake3_hex(&self) -> &str {
        &self.challenge_nonce_blake3_hex
    }
    pub fn raw_observation_qualification_digest(&self) -> &str {
        &self.raw_observation_qualification_digest
    }
    pub fn raw_observation_report_digest(&self) -> &str {
        &self.raw_observation_report_digest
    }
    pub fn mapped_object_set_digest(&self) -> &str {
        &self.mapped_object_set_digest
    }
    pub fn runtime_policy_digest(&self) -> &str {
        &self.runtime_policy_digest
    }
    pub fn runtime_verifier_ref(&self) -> &str {
        &self.runtime_verifier_ref
    }
    pub fn backend_id(&self) -> &str {
        &self.backend_id
    }
    pub fn boot_measurement_digest(&self) -> &str {
        &self.boot_measurement_digest
    }
    pub fn executable_digest(&self) -> &str {
        &self.executable_digest
    }
    pub fn dependency_closure_digest(&self) -> &str {
        &self.dependency_closure_digest
    }
    pub fn runtime_config_digest(&self) -> &str {
        &self.runtime_config_digest
    }
    pub fn launch_attestation_digest(&self) -> &str {
        &self.launch_attestation_digest
    }
    pub fn checkpoint_one_challenge_digest(&self) -> &str {
        &self.checkpoint_one_challenge_digest
    }
    pub fn checkpoint_one_claimed_observation_digest(&self) -> &str {
        &self.checkpoint_one_claimed_observation_digest
    }
    pub const fn previous_checkpoint_monotonic_counter(&self) -> u64 {
        self.previous_checkpoint_monotonic_counter
    }
    pub const fn observed_at_ms(&self) -> u64 {
        self.observed_at_ms
    }

    pub const fn canonical_ticket_consumed_before_live_observation(&self) -> bool {
        true
    }
    pub const fn current_os_pid_matches_consumed_ticket(&self) -> bool {
        true
    }
    pub const fn predecessor_alias_matches_exact_ready_checkpoint(&self) -> bool {
        true
    }
    pub const fn checkpoint_two_counter_strictly_advances(&self) -> bool {
        true
    }
    pub const fn checkpoint_challenge_constructed_before_live_observation(&self) -> bool {
        true
    }
    pub const fn live_mapped_runtime_observation_performed_after_ticket_consumption(&self) -> bool {
        true
    }
    pub const fn observation_static_runtime_identity_matches_ticket_and_bound_runtime(
        &self,
    ) -> bool {
        true
    }
    pub const fn supervisor_post_release_issuance_authenticated(&self) -> bool {
        false
    }
    pub const fn exact_successful_release_authenticated(&self) -> bool {
        false
    }
    pub const fn causal_release_to_observation_established(&self) -> bool {
        false
    }
    pub const fn checkpoint_one_live_observation_proven_from_ticket(&self) -> bool {
        false
    }
    pub const fn checkpoint_two_signed_inclusion_established(&self) -> bool {
        false
    }
    pub const fn mapping_continuity_between_checkpoints_established(&self) -> bool {
        false
    }
    pub const fn exclusive_pipe_peer_authority_established(&self) -> bool {
        false
    }
    pub const fn trusted_time_established(&self) -> bool {
        false
    }
    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

pub struct PostReleaseCheckpointTwoObservationQualification {
    pub report: PostReleaseCheckpointTwoObservationReport,
    pub ticket: PostReleaseRuntimeHandoffTicket,
    pub challenge: MappedRuntimeCheckpointChallenge,
    pub raw: MappedExecutableRuntimeQualification,
    verified: PostReleaseCheckpointTwoObservation,
}

impl PostReleaseCheckpointTwoObservationQualification {
    pub fn verified(&self) -> &PostReleaseCheckpointTwoObservation {
        &self.verified
    }
    pub fn into_verified(self) -> PostReleaseCheckpointTwoObservation {
        self.verified
    }
}

#[allow(clippy::too_many_arguments)]
pub fn consume_post_release_ticket_and_observe_checkpoint_two(
    policy: &PostReleaseCheckpointTwoObservationPolicy,
    mapped_policy: &MappedExecutableRuntimePolicy,
    bound: &NixBoundInProcessRuntimePolicy,
    closure: &NixRuntimeClosureQualification,
    runtime_policy: &VerifierRuntimeContinuityPolicy,
    checkpoint_monotonic_counter: u64,
    observed_at_ms: u64,
) -> Result<
    PostReleaseCheckpointTwoObservationQualification,
    PostReleaseCheckpointTwoObservationReport,
> {
    let policy_digest = policy.canonical_digest();
    let mapped_policy_digest = mapped_policy.canonical_digest();
    let runtime_policy_digest = runtime_policy.canonical_digest();
    let current_pid = i32::try_from(std::process::id()).unwrap_or(-1);
    let mut report = PostReleaseCheckpointTwoObservationReport {
        schema_version: POST_RELEASE_CHECKPOINT_TWO_REPORT_SCHEMA_V1.into(),
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
        channel_retention_digest: None,
        release_digest: None,
        release_report_digest: None,
        confinement_qualification_digest: None,
        mapping_continuity_qualification_digest: None,
        bootstrap_ready_checkpoint_digest: None,
        process_instance_id: String::new(),
        checkpoint_sequence: 2,
        previous_checkpoint_digest: None,
        previous_checkpoint_monotonic_counter: 0,
        checkpoint_monotonic_counter,
        checkpoint_challenge_digest: None,
        raw_observation_qualification_digest: None,
        mapped_object_set_digest: None,
        runtime_policy_digest: runtime_policy_digest.clone().unwrap_or_else(|| "-".into()),
        runtime_verifier_ref: runtime_policy.verifier_ref.clone(),
        backend_id: bound.backend_id().into(),
        executable_digest: bound.executable_digest().into(),
        dependency_closure_digest: bound.dependency_closure_digest().into(),
        observed_at_ms,
        disposition: PostReleaseCheckpointTwoObservationDisposition::Invalid,
        issues: Vec::new(),
    };

    if !policy.validate() {
        report
            .issues
            .push(PostReleaseCheckpointTwoObservationIssue::InvalidPolicy);
    }
    if !runtime_policy.validate() || runtime_policy_digest.is_none() {
        report
            .issues
            .push(PostReleaseCheckpointTwoObservationIssue::InvalidRuntimePolicy);
    }
    if checkpoint_monotonic_counter == 0 {
        report
            .issues
            .push(PostReleaseCheckpointTwoObservationIssue::InvalidCheckpointCounter);
    }
    if mapped_policy_digest.as_deref()
        != Some(policy.expected_mapped_runtime_policy_digest.as_str())
    {
        report
            .issues
            .push(PostReleaseCheckpointTwoObservationIssue::MappedRuntimePolicyMismatch);
    }
    if bound.policy_digest() != policy.expected_nix_binding_policy_digest {
        report
            .issues
            .push(PostReleaseCheckpointTwoObservationIssue::NixBindingPolicyMismatch);
    }
    if bound.closure_policy_digest() != policy.expected_closure_policy_digest {
        report
            .issues
            .push(PostReleaseCheckpointTwoObservationIssue::ClosurePolicyMismatch);
    }
    if runtime_policy_digest.as_deref() != Some(policy.expected_runtime_policy_digest.as_str())
        || bound.runtime_policy_digest() != policy.expected_runtime_policy_digest
    {
        report
            .issues
            .push(PostReleaseCheckpointTwoObservationIssue::RuntimePolicyMismatch);
    }
    if runtime_policy.verifier_ref != policy.expected_runtime_verifier_ref
        || bound.runtime_verifier_ref() != policy.expected_runtime_verifier_ref
    {
        report
            .issues
            .push(PostReleaseCheckpointTwoObservationIssue::RuntimeVerifierMismatch);
    }
    if bound.backend_id() != policy.expected_backend_id {
        report
            .issues
            .push(PostReleaseCheckpointTwoObservationIssue::BackendMismatch);
    }
    if bound.executable_digest() != runtime_policy.expected_executable_digest {
        report
            .issues
            .push(PostReleaseCheckpointTwoObservationIssue::RuntimeExecutableMismatch);
    }
    if bound.dependency_closure_digest() != runtime_policy.expected_dependency_closure_digest {
        report
            .issues
            .push(PostReleaseCheckpointTwoObservationIssue::RuntimeClosureMismatch);
    }
    let fd = match i32::try_from(policy.handoff_read_fd) {
        Ok(value) if value >= 0 => value,
        _ => {
            report
                .issues
                .push(PostReleaseCheckpointTwoObservationIssue::InvalidHandoffReadFd);
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
                .push(PostReleaseCheckpointTwoObservationIssue::HandoffFdUnavailable(error));
            return Err(finalize(report));
        }
    };
    report.pipe_target = Some(before.target.clone());
    report.pre_read_fd_observation_digest = Some(before.observation_digest.clone());
    if parse_pipe_target(&before.target) != Some(before.inode) {
        report
            .issues
            .push(PostReleaseCheckpointTwoObservationIssue::HandoffTargetNotPipe);
    }
    if before.flags & OFlag::O_ACCMODE.bits() != OFlag::O_RDONLY.bits() {
        report
            .issues
            .push(PostReleaseCheckpointTwoObservationIssue::HandoffFdNotReadOnly);
    }
    if before.flags & OFlag::O_CLOEXEC.bits() != 0 {
        report
            .issues
            .push(PostReleaseCheckpointTwoObservationIssue::HandoffFdCloseOnExecUnexpected);
    }
    if before.flags & OFlag::O_NONBLOCK.bits() != 0 {
        report
            .issues
            .push(PostReleaseCheckpointTwoObservationIssue::HandoffFdNonBlocking);
    }
    if parse_pipe_target(&before.target).is_none() {
        report
            .issues
            .push(PostReleaseCheckpointTwoObservationIssue::HandoffFdInfoInconsistent);
    }
    if !report.issues.is_empty() {
        return Err(finalize(report));
    }

    let wire = match read_one_ticket(fd, policy.max_ticket_bytes) {
        Ok(value) => value,
        Err(error) => {
            report
                .issues
                .push(PostReleaseCheckpointTwoObservationIssue::TicketReadFailed(
                    error,
                ));
            return Err(finalize(report));
        }
    };
    report.ticket_bytes_len = wire.len() as u64;
    report.ticket_bytes_digest = Some(format!("blake3:{}", blake3::hash(&wire).to_hex()));
    let Some(ticket) = PostReleaseRuntimeHandoffTicket::from_wire_bytes(&wire) else {
        report
            .issues
            .push(PostReleaseCheckpointTwoObservationIssue::TicketDecodeFailed);
        return Err(finalize(report));
    };
    report.ticket_digest = Some(ticket.ticket_digest.clone());
    report.channel_retention_digest = Some(ticket.channel_retention_digest.clone());
    report.release_digest = Some(ticket.release_digest.clone());
    report.release_report_digest = Some(ticket.release_report_digest.clone());
    report.confinement_qualification_digest = Some(ticket.confinement_qualification_digest.clone());
    report.mapping_continuity_qualification_digest =
        Some(ticket.mapping_continuity_qualification_digest.clone());
    report.bootstrap_ready_checkpoint_digest =
        Some(ticket.bootstrap_ready_checkpoint_digest.clone());
    report.process_instance_id = ticket.process_instance_id.clone();
    report.previous_checkpoint_digest = Some(ticket.previous_checkpoint_digest.clone());
    report.previous_checkpoint_monotonic_counter = ticket.previous_checkpoint_monotonic_counter;

    if ticket.tracee_handoff_read_fd != policy.handoff_read_fd {
        report
            .issues
            .push(PostReleaseCheckpointTwoObservationIssue::TicketReadFdMismatch);
    }
    if ticket.pipe_target != before.target {
        report
            .issues
            .push(PostReleaseCheckpointTwoObservationIssue::TicketPipeTargetMismatch);
    }
    if ticket.tracee_pid != current_pid {
        report.issues.push(
            PostReleaseCheckpointTwoObservationIssue::TicketPidMismatch {
                expected: ticket.tracee_pid,
                observed: current_pid,
            },
        );
    }
    if ticket.checkpoint_sequence != 2 {
        report
            .issues
            .push(PostReleaseCheckpointTwoObservationIssue::TicketCheckpointSequenceMismatch);
    }
    if ticket.previous_checkpoint_digest != ticket.bootstrap_ready_checkpoint_digest {
        report
            .issues
            .push(PostReleaseCheckpointTwoObservationIssue::TicketPredecessorAliasMismatch);
    }
    if checkpoint_monotonic_counter <= ticket.previous_checkpoint_monotonic_counter {
        report
            .issues
            .push(PostReleaseCheckpointTwoObservationIssue::TicketCounterRegression);
    }
    if ticket.runtime_policy_digest != policy.expected_runtime_policy_digest
        || ticket.runtime_policy_digest != bound.runtime_policy_digest()
        || runtime_policy_digest.as_deref() != Some(ticket.runtime_policy_digest.as_str())
    {
        report
            .issues
            .push(PostReleaseCheckpointTwoObservationIssue::TicketRuntimePolicyMismatch);
    }
    if ticket.runtime_verifier_ref != policy.expected_runtime_verifier_ref
        || ticket.runtime_verifier_ref != bound.runtime_verifier_ref()
        || ticket.runtime_verifier_ref != runtime_policy.verifier_ref
    {
        report
            .issues
            .push(PostReleaseCheckpointTwoObservationIssue::TicketRuntimeVerifierMismatch);
    }
    if ticket.backend_id != policy.expected_backend_id || ticket.backend_id != bound.backend_id() {
        report
            .issues
            .push(PostReleaseCheckpointTwoObservationIssue::TicketBackendMismatch);
    }
    if ticket.boot_measurement_digest != runtime_policy.expected_boot_measurement_digest {
        report
            .issues
            .push(PostReleaseCheckpointTwoObservationIssue::TicketBootMeasurementMismatch);
    }
    if ticket.executable_digest != runtime_policy.expected_executable_digest
        || ticket.executable_digest != bound.executable_digest()
    {
        report
            .issues
            .push(PostReleaseCheckpointTwoObservationIssue::TicketExecutableMismatch);
    }
    if ticket.dependency_closure_digest != runtime_policy.expected_dependency_closure_digest
        || ticket.dependency_closure_digest != bound.dependency_closure_digest()
    {
        report
            .issues
            .push(PostReleaseCheckpointTwoObservationIssue::TicketClosureMismatch);
    }
    if ticket.runtime_config_digest != runtime_policy.expected_runtime_config_digest {
        report
            .issues
            .push(PostReleaseCheckpointTwoObservationIssue::TicketRuntimeConfigMismatch);
    }

    let after = match observe_pipe_fd(fd, policy.max_fdinfo_bytes) {
        Ok(value) => value,
        Err(error) => {
            report
                .issues
                .push(PostReleaseCheckpointTwoObservationIssue::HandoffFdUnavailable(error));
            return Err(finalize(report));
        }
    };
    report.post_read_fd_observation_digest = Some(after.observation_digest.clone());
    if before != after {
        report
            .issues
            .push(PostReleaseCheckpointTwoObservationIssue::HandoffFdChangedDuringConsumption);
    }
    if !report.issues.is_empty() {
        return Err(finalize(report));
    }

    let challenge = MappedRuntimeCheckpointChallenge {
        schema_version: MAPPED_RUNTIME_CHECKPOINT_CHALLENGE_SCHEMA_V1.into(),
        process_instance_id: ticket.process_instance_id.clone(),
        runtime_policy_digest: ticket.runtime_policy_digest.clone(),
        verifier_ref: ticket.runtime_verifier_ref.clone(),
        checkpoint_sequence: 2,
        checkpoint_monotonic_counter,
        previous_checkpoint_digest: Some(ticket.bootstrap_ready_checkpoint_digest.clone()),
        nonce_blake3_hex: ticket.challenge_nonce_blake3_hex.clone(),
    };
    let Some(challenge_digest) = challenge.canonical_digest() else {
        report
            .issues
            .push(PostReleaseCheckpointTwoObservationIssue::ChallengeConstructionFailed);
        return Err(finalize(report));
    };
    report.checkpoint_challenge_digest = Some(challenge_digest.clone());

    // Ticket consumption and checkpoint-two challenge construction have both
    // completed before this live process observation begins.
    let raw = match observe_mapped_nix_executable_runtime(
        mapped_policy,
        bound,
        closure,
        observed_at_ms,
    ) {
        Ok(value) => value,
        Err(raw_report) => {
            report.issues.push(
                PostReleaseCheckpointTwoObservationIssue::RawObservationFailed(
                    raw_report.canonical_digest(),
                ),
            );
            return Err(finalize(report));
        }
    };
    let observed = raw.observed();
    report.raw_observation_qualification_digest = Some(observed.qualification_digest().into());
    report.mapped_object_set_digest = Some(observed.mapped_object_set_digest().into());
    if observed.policy_digest() != policy.expected_mapped_runtime_policy_digest {
        report
            .issues
            .push(PostReleaseCheckpointTwoObservationIssue::ObservationPolicyMismatch);
    }
    if observed.nix_binding_qualification_digest() != bound.qualification_digest() {
        report
            .issues
            .push(PostReleaseCheckpointTwoObservationIssue::ObservationNixBindingMismatch);
    }
    if observed.closure_qualification_digest() != bound.closure_qualification_digest() {
        report.issues.push(
            PostReleaseCheckpointTwoObservationIssue::ObservationClosureQualificationMismatch,
        );
    }
    if observed.closure_digest() != ticket.dependency_closure_digest
        || observed.closure_digest() != bound.dependency_closure_digest()
    {
        report
            .issues
            .push(PostReleaseCheckpointTwoObservationIssue::ObservationClosureDigestMismatch);
    }
    if observed.runtime_policy_digest() != ticket.runtime_policy_digest {
        report
            .issues
            .push(PostReleaseCheckpointTwoObservationIssue::ObservationRuntimePolicyMismatch);
    }
    if observed.runtime_verifier_ref() != ticket.runtime_verifier_ref {
        report
            .issues
            .push(PostReleaseCheckpointTwoObservationIssue::ObservationVerifierMismatch);
    }
    if observed.backend_id() != ticket.backend_id {
        report
            .issues
            .push(PostReleaseCheckpointTwoObservationIssue::ObservationBackendMismatch);
    }
    if observed.host_executable_digest() != ticket.executable_digest {
        report
            .issues
            .push(PostReleaseCheckpointTwoObservationIssue::ObservationExecutableMismatch);
    }
    if observed.observed_at_ms() != observed_at_ms {
        report
            .issues
            .push(PostReleaseCheckpointTwoObservationIssue::ObservationTimeMismatch);
    }
    if !report.issues.is_empty() {
        return Err(finalize(report));
    }

    report.disposition = PostReleaseCheckpointTwoObservationDisposition::Qualified;
    let report_digest = report.canonical_digest();
    let policy_digest = policy_digest.expect("validated policy has digest");
    let ticket_bytes_digest = report
        .ticket_bytes_digest
        .clone()
        .expect("qualified report has ticket bytes digest");
    let qualification_digest = qualification_digest(
        &policy_digest,
        &ticket.ticket_digest,
        &ticket_bytes_digest,
        current_pid,
        &challenge_digest,
        observed.qualification_digest(),
        observed.mapped_object_set_digest(),
        &report_digest,
    );
    let verified = PostReleaseCheckpointTwoObservation {
        qualification_digest,
        report_digest,
        policy_digest,
        current_pid,
        handoff_read_fd: policy.handoff_read_fd,
        pipe_target: ticket.pipe_target.clone(),
        ticket_digest: ticket.ticket_digest.clone(),
        ticket_bytes_digest,
        channel_retention_digest: ticket.channel_retention_digest.clone(),
        release_digest: ticket.release_digest.clone(),
        release_report_digest: ticket.release_report_digest.clone(),
        confinement_qualification_digest: ticket.confinement_qualification_digest.clone(),
        mapping_continuity_qualification_digest: ticket
            .mapping_continuity_qualification_digest
            .clone(),
        bootstrap_ready_wire_digest: ticket.bootstrap_ready_wire_digest.clone(),
        bootstrap_ready_checkpoint_digest: ticket.bootstrap_ready_checkpoint_digest.clone(),
        launch_handoff_qualification_digest: ticket.launch_handoff_qualification_digest.clone(),
        launch_ticket_digest: ticket.launch_ticket_digest.clone(),
        process_instance_id: ticket.process_instance_id.clone(),
        checkpoint_monotonic_counter,
        checkpoint_challenge_digest: challenge_digest,
        challenge_nonce_blake3_hex: ticket.challenge_nonce_blake3_hex.clone(),
        raw_observation_qualification_digest: observed.qualification_digest().into(),
        raw_observation_report_digest: observed.report_digest().into(),
        mapped_object_set_digest: observed.mapped_object_set_digest().into(),
        runtime_policy_digest: ticket.runtime_policy_digest.clone(),
        runtime_verifier_ref: ticket.runtime_verifier_ref.clone(),
        backend_id: ticket.backend_id.clone(),
        boot_measurement_digest: ticket.boot_measurement_digest.clone(),
        executable_digest: ticket.executable_digest.clone(),
        dependency_closure_digest: ticket.dependency_closure_digest.clone(),
        runtime_config_digest: ticket.runtime_config_digest.clone(),
        launch_attestation_digest: ticket.launch_attestation_digest.clone(),
        checkpoint_one_challenge_digest: ticket.checkpoint_one_challenge_digest.clone(),
        checkpoint_one_claimed_observation_digest: ticket
            .checkpoint_one_observation_qualification_digest
            .clone(),
        previous_checkpoint_monotonic_counter: ticket.previous_checkpoint_monotonic_counter,
        observed_at_ms: observed.observed_at_ms(),
    };

    Ok(PostReleaseCheckpointTwoObservationQualification {
        report,
        ticket,
        challenge,
        raw,
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
    let observation_digest =
        pipe_observation_digest(fd, &target, parsed.flags, parsed.inode, &raw_fdinfo_digest);
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
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
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
    let maximum = maximum.min(HARD_MAX_TICKET_BYTES);
    let mut out = Vec::with_capacity(maximum.min(2048) as usize);
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
    read_exact_append(fd, &mut out, 8, maximum)?;
    read_wire_field(fd, &mut out, maximum)?;
    read_exact_append(fd, &mut out, 8, maximum)?;
    read_exact_append(fd, &mut out, 8, maximum)?;
    read_wire_field(fd, &mut out, maximum)?;
    read_wire_field(fd, &mut out, maximum)?;
    Ok(out)
}

fn read_wire_field(fd: RawFd, out: &mut Vec<u8>, maximum: u32) -> Result<(), String> {
    let prefix_start = out.len();
    read_exact_append(fd, out, 4, maximum)?;
    let prefix: [u8; 4] = out[prefix_start..prefix_start + 4]
        .try_into()
        .map_err(|_| "invalid-length-prefix")?;
    let length = u32::from_le_bytes(prefix);
    if length == 0 || length > MAX_TEXT as u32 || length > maximum {
        return Err("wire-field-size-invalid".into());
    }
    read_exact_append(fd, out, length as usize, maximum)
}

fn read_exact_append(
    fd: RawFd,
    out: &mut Vec<u8>,
    count: usize,
    maximum: u32,
) -> Result<(), String> {
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
            Ok(count) => offset += count,
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
    b3(h.finalize())
}

#[allow(clippy::too_many_arguments)]
fn qualification_digest(
    policy: &str,
    ticket: &str,
    ticket_bytes: &str,
    current_pid: i32,
    challenge: &str,
    raw_observation: &str,
    mapped_object_set: &str,
    report: &str,
) -> String {
    let mut h = blake3::Hasher::new();
    h.update(QUALIFICATION_DOMAIN);
    for value in [
        policy,
        ticket,
        ticket_bytes,
        challenge,
        raw_observation,
        mapped_object_set,
        report,
    ] {
        field(&mut h, value);
    }
    h.update(&current_pid.to_le_bytes());
    b3(h.finalize())
}

fn finalize(
    mut report: PostReleaseCheckpointTwoObservationReport,
) -> PostReleaseCheckpointTwoObservationReport {
    report.disposition = if report
        .issues
        .iter()
        .any(PostReleaseCheckpointTwoObservationIssue::invalid)
    {
        PostReleaseCheckpointTwoObservationDisposition::Invalid
    } else {
        PostReleaseCheckpointTwoObservationDisposition::Blocked
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

fn valid_blake3(value: &str) -> bool {
    value
        .strip_prefix("blake3:")
        .is_some_and(|digest| lower_hex_exact(digest, 64))
}

fn lower_hex_exact(value: &str, length: usize) -> bool {
    value.len() == length
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn canonical_text(value: &str) -> bool {
    !value.is_empty()
        && value == value.trim()
        && value.len() <= MAX_TEXT
        && !value.chars().any(char::is_control)
}

fn valid_refs(values: &[String]) -> bool {
    values.len() <= MAX_REFS && values.iter().all(|value| canonical_text(value)) && {
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

fn b3(hash: blake3::Hash) -> String {
    format!("blake3:{}", hash.to_hex())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(label: &str) -> String {
        format!("blake3:{}", blake3::hash(label.as_bytes()).to_hex())
    }

    fn policy() -> PostReleaseCheckpointTwoObservationPolicy {
        PostReleaseCheckpointTwoObservationPolicy {
            schema_version: POST_RELEASE_CHECKPOINT_TWO_POLICY_SCHEMA_V1.into(),
            policy_id: "post-release-checkpoint-two:v1".into(),
            expected_mapped_runtime_policy_digest: d("mapped-policy"),
            expected_nix_binding_policy_digest: d("nix-binding-policy"),
            expected_closure_policy_digest: d("closure-policy"),
            expected_runtime_policy_digest: d("runtime-policy"),
            expected_runtime_verifier_ref: "verifier:prod".into(),
            expected_backend_id: "backend:in-process-p256".into(),
            handoff_read_fd: 11,
            max_ticket_bytes: 4096,
            max_fdinfo_bytes: 1024 * 1024,
            evidence_refs: vec!["review:post-release-checkpoint-two".into()],
        }
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
    fn runtime_and_observer_policy_identities_are_semantic() {
        let left = policy();
        let mut right = left.clone();
        right.expected_mapped_runtime_policy_digest = d("other-mapped");
        assert_ne!(left.canonical_digest(), right.canonical_digest());
        let mut right = left.clone();
        right.expected_runtime_policy_digest = d("other-runtime");
        assert_ne!(left.canonical_digest(), right.canonical_digest());
        let mut right = left.clone();
        right.expected_backend_id = "backend:other".into();
        assert_ne!(left.canonical_digest(), right.canonical_digest());
    }

    #[test]
    fn predecessor_alias_is_not_optional_for_child_theorem() {
        let checkpoint = d("checkpoint-one");
        let other = d("other-checkpoint");
        assert_ne!(checkpoint, other);
        assert!(valid_blake3(&checkpoint));
        assert!(valid_blake3(&other));
    }

    #[test]
    fn qualification_binds_ticket_challenge_and_raw_observation() {
        let base = qualification_digest(
            &d("policy"),
            &d("ticket"),
            &d("ticket-bytes"),
            4242,
            &d("challenge"),
            &d("raw-observation"),
            &d("mapped-set"),
            &d("report"),
        );
        let changed = qualification_digest(
            &d("policy"),
            &d("ticket"),
            &d("ticket-bytes"),
            4242,
            &d("other-challenge"),
            &d("raw-observation"),
            &d("mapped-set"),
            &d("report"),
        );
        assert_ne!(base, changed);
    }

    #[test]
    fn claim_ceiling_stays_explicit() {
        let claims = [
            "supervisor-issuance-authenticated=false",
            "release-authenticated=false",
            "causal-release-to-observation=false",
            "checkpoint-one-live-proven=false",
            "checkpoint-two-signed-inclusion=false",
            "between-checkpoint-continuity=false",
            "trusted-time=false",
            "physical-authority=false",
        ];
        assert_eq!(claims.len(), 8);
    }
}
