// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exact composition of supervisor-issued post-release re-entry and tracee-side
//! checkpoint-two live mapped-runtime observation.
//!
//! This bridge adds no new observation, pipe, entropy or signature mechanism.
//! Its only authority is exact anti-splice composition of two opaque parent
//! capabilities across the same canonical ticket bytes.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use symthaea_assurance_post_release_checkpoint_two_observation::
    PostReleaseCheckpointTwoObservation;
use symthaea_assurance_post_release_runtime_handoff::IssuedPostReleaseRuntimeChallenge;

pub const AUTHENTICATED_POST_RELEASE_CHECKPOINT_TWO_POLICY_SCHEMA_V1: &str =
    "symthaea.assurance.authenticated-post-release-checkpoint-two-policy.v1";
pub const AUTHENTICATED_POST_RELEASE_CHECKPOINT_TWO_REPORT_SCHEMA_V1: &str =
    "symthaea.assurance.authenticated-post-release-checkpoint-two-report.v1";

const POLICY_DOMAIN: &[u8] =
    b"symthaea.assurance.authenticated-post-release-checkpoint-two-policy.digest.v1\0";
const REPORT_DOMAIN: &[u8] =
    b"symthaea.assurance.authenticated-post-release-checkpoint-two-report.digest.v1\0";
const QUALIFICATION_DOMAIN: &[u8] =
    b"symthaea.assurance.authenticated-post-release-checkpoint-two-qualification.digest.v1\0";
const MAX_TEXT: usize = 4096;
const MAX_REFS: usize = 128;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthenticatedPostReleaseCheckpointTwoPolicy {
    pub schema_version: String,
    pub policy_id: String,
    pub expected_issuer_policy_digest: String,
    pub expected_tracee_observation_policy_digest: String,
    pub expected_runtime_policy_digest: String,
    pub expected_runtime_verifier_ref: String,
    pub expected_backend_id: String,
    pub evidence_refs: Vec<String>,
}

impl AuthenticatedPostReleaseCheckpointTwoPolicy {
    pub fn validate(&self) -> bool {
        self.schema_version == AUTHENTICATED_POST_RELEASE_CHECKPOINT_TWO_POLICY_SCHEMA_V1
            && canonical_text(&self.policy_id)
            && valid_blake3(&self.expected_issuer_policy_digest)
            && valid_blake3(&self.expected_tracee_observation_policy_digest)
            && valid_blake3(&self.expected_runtime_policy_digest)
            && canonical_text(&self.expected_runtime_verifier_ref)
            && canonical_text(&self.expected_backend_id)
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
            self.expected_issuer_policy_digest.as_str(),
            self.expected_tracee_observation_policy_digest.as_str(),
            self.expected_runtime_policy_digest.as_str(),
            self.expected_runtime_verifier_ref.as_str(),
            self.expected_backend_id.as_str(),
        ] {
            field(&mut h, value);
        }
        sorted(&mut h, &self.evidence_refs);
        Some(b3(h.finalize()))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuthenticatedPostReleaseCheckpointTwoIssue {
    InvalidPolicy,
    IssuerPolicyMismatch,
    ObservationPolicyMismatch,
    TicketDigestMismatch,
    TicketBytesMismatch,
    RetentionMismatch,
    ReleaseMismatch,
    ReleaseReportMismatch,
    ConfinementMismatch,
    MappingContinuityMismatch,
    BootstrapReadyWireMismatch,
    BootstrapReadyCheckpointMismatch,
    LaunchHandoffMismatch,
    LaunchTicketMismatch,
    TraceePidMismatch,
    HandoffReadFdMismatch,
    PipeTargetMismatch,
    ProcessInstanceMismatch,
    RuntimePolicyMismatch,
    RuntimeVerifierMismatch,
    BackendMismatch,
    BootMeasurementMismatch,
    ExecutableMismatch,
    DependencyClosureMismatch,
    RuntimeConfigMismatch,
    LaunchAttestationMismatch,
    CheckpointOneChallengeMismatch,
    CheckpointOneClaimedObservationMismatch,
    CheckpointSequenceMismatch,
    PreviousCheckpointMismatch,
    PreviousCounterMismatch,
    ChallengeNonceMismatch,
}

impl AuthenticatedPostReleaseCheckpointTwoIssue {
    fn invalid(&self) -> bool {
        matches!(
            self,
            Self::InvalidPolicy
                | Self::TicketDigestMismatch
                | Self::TicketBytesMismatch
                | Self::CheckpointSequenceMismatch
                | Self::PreviousCheckpointMismatch
                | Self::ChallengeNonceMismatch
        )
    }

    fn code(&self) -> &'static str {
        match self {
            Self::InvalidPolicy => "invalid-policy",
            Self::IssuerPolicyMismatch => "issuer-policy-mismatch",
            Self::ObservationPolicyMismatch => "observation-policy-mismatch",
            Self::TicketDigestMismatch => "ticket-digest-mismatch",
            Self::TicketBytesMismatch => "ticket-bytes-mismatch",
            Self::RetentionMismatch => "retention-mismatch",
            Self::ReleaseMismatch => "release-mismatch",
            Self::ReleaseReportMismatch => "release-report-mismatch",
            Self::ConfinementMismatch => "confinement-mismatch",
            Self::MappingContinuityMismatch => "mapping-continuity-mismatch",
            Self::BootstrapReadyWireMismatch => "bootstrap-ready-wire-mismatch",
            Self::BootstrapReadyCheckpointMismatch => "bootstrap-ready-checkpoint-mismatch",
            Self::LaunchHandoffMismatch => "launch-handoff-mismatch",
            Self::LaunchTicketMismatch => "launch-ticket-mismatch",
            Self::TraceePidMismatch => "tracee-pid-mismatch",
            Self::HandoffReadFdMismatch => "handoff-read-fd-mismatch",
            Self::PipeTargetMismatch => "pipe-target-mismatch",
            Self::ProcessInstanceMismatch => "process-instance-mismatch",
            Self::RuntimePolicyMismatch => "runtime-policy-mismatch",
            Self::RuntimeVerifierMismatch => "runtime-verifier-mismatch",
            Self::BackendMismatch => "backend-mismatch",
            Self::BootMeasurementMismatch => "boot-measurement-mismatch",
            Self::ExecutableMismatch => "executable-mismatch",
            Self::DependencyClosureMismatch => "dependency-closure-mismatch",
            Self::RuntimeConfigMismatch => "runtime-config-mismatch",
            Self::LaunchAttestationMismatch => "launch-attestation-mismatch",
            Self::CheckpointOneChallengeMismatch => "checkpoint-one-challenge-mismatch",
            Self::CheckpointOneClaimedObservationMismatch => {
                "checkpoint-one-claimed-observation-mismatch"
            }
            Self::CheckpointSequenceMismatch => "checkpoint-sequence-mismatch",
            Self::PreviousCheckpointMismatch => "previous-checkpoint-mismatch",
            Self::PreviousCounterMismatch => "previous-counter-mismatch",
            Self::ChallengeNonceMismatch => "challenge-nonce-mismatch",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AuthenticatedPostReleaseCheckpointTwoDisposition {
    Invalid,
    Blocked,
    Qualified,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AuthenticatedPostReleaseCheckpointTwoReport {
    pub schema_version: String,
    pub policy_id: String,
    pub policy_digest: Option<String>,
    pub issuer_qualification_digest: String,
    pub observation_qualification_digest: String,
    pub ticket_digest: String,
    pub ticket_bytes_digest: String,
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
    pub handoff_read_fd: u32,
    pub pipe_target: String,
    pub process_instance_id: String,
    pub runtime_policy_digest: String,
    pub runtime_verifier_ref: String,
    pub backend_id: String,
    pub executable_digest: String,
    pub dependency_closure_digest: String,
    pub checkpoint_two_challenge_digest: String,
    pub checkpoint_two_observation_qualification_digest: String,
    pub checkpoint_two_mapped_object_set_digest: String,
    pub checkpoint_two_counter: u64,
    pub previous_checkpoint_counter: u64,
    pub checkpoint_two_observed_at_ms: u64,
    pub disposition: AuthenticatedPostReleaseCheckpointTwoDisposition,
    pub issues: Vec<AuthenticatedPostReleaseCheckpointTwoIssue>,
}

impl AuthenticatedPostReleaseCheckpointTwoReport {
    pub fn canonical_digest(&self) -> String {
        let mut h = blake3::Hasher::new();
        h.update(REPORT_DOMAIN);
        for value in [
            self.schema_version.as_str(),
            self.policy_id.as_str(),
            self.policy_digest.as_deref().unwrap_or("-"),
            self.issuer_qualification_digest.as_str(),
            self.observation_qualification_digest.as_str(),
            self.ticket_digest.as_str(),
            self.ticket_bytes_digest.as_str(),
            self.channel_retention_digest.as_str(),
            self.release_digest.as_str(),
            self.release_report_digest.as_str(),
            self.confinement_qualification_digest.as_str(),
            self.mapping_continuity_qualification_digest.as_str(),
            self.bootstrap_ready_wire_digest.as_str(),
            self.bootstrap_ready_checkpoint_digest.as_str(),
            self.launch_handoff_qualification_digest.as_str(),
            self.launch_ticket_digest.as_str(),
            self.pipe_target.as_str(),
            self.process_instance_id.as_str(),
            self.runtime_policy_digest.as_str(),
            self.runtime_verifier_ref.as_str(),
            self.backend_id.as_str(),
            self.executable_digest.as_str(),
            self.dependency_closure_digest.as_str(),
            self.checkpoint_two_challenge_digest.as_str(),
            self.checkpoint_two_observation_qualification_digest.as_str(),
            self.checkpoint_two_mapped_object_set_digest.as_str(),
        ] {
            field(&mut h, value);
        }
        h.update(&self.tracee_pid.to_le_bytes());
        h.update(&self.handoff_read_fd.to_le_bytes());
        h.update(&self.checkpoint_two_counter.to_le_bytes());
        h.update(&self.previous_checkpoint_counter.to_le_bytes());
        h.update(&self.checkpoint_two_observed_at_ms.to_le_bytes());
        field(
            &mut h,
            match self.disposition {
                AuthenticatedPostReleaseCheckpointTwoDisposition::Invalid => "invalid",
                AuthenticatedPostReleaseCheckpointTwoDisposition::Blocked => "blocked",
                AuthenticatedPostReleaseCheckpointTwoDisposition::Qualified => "qualified",
            },
        );
        h.update(&(self.issues.len() as u64).to_le_bytes());
        for issue in &self.issues {
            field(&mut h, issue.code());
        }
        b3(h.finalize())
    }

    pub const fn grants_physical_authority(&self) -> bool { false }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AuthenticatedPostReleaseCheckpointTwoObservation {
    qualification_digest: String,
    report_digest: String,
    policy_digest: String,
    issuer_qualification_digest: String,
    tracee_observation_qualification_digest: String,
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
    tracee_pid: i32,
    handoff_read_fd: u32,
    pipe_target: String,
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
    checkpoint_one_claimed_observation_digest: String,
    previous_checkpoint_counter: u64,
    checkpoint_two_counter: u64,
    checkpoint_two_challenge_digest: String,
    checkpoint_two_challenge_nonce_blake3_hex: String,
    checkpoint_two_observation_qualification_digest: String,
    checkpoint_two_observation_report_digest: String,
    checkpoint_two_mapped_object_set_digest: String,
    checkpoint_two_observed_at_ms: u64,
}

impl AuthenticatedPostReleaseCheckpointTwoObservation {
    pub fn qualification_digest(&self) -> &str { &self.qualification_digest }
    pub fn report_digest(&self) -> &str { &self.report_digest }
    pub fn policy_digest(&self) -> &str { &self.policy_digest }
    pub fn issuer_qualification_digest(&self) -> &str { &self.issuer_qualification_digest }
    pub fn tracee_observation_qualification_digest(&self) -> &str {
        &self.tracee_observation_qualification_digest
    }
    pub fn ticket_digest(&self) -> &str { &self.ticket_digest }
    pub fn ticket_bytes_digest(&self) -> &str { &self.ticket_bytes_digest }
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
    pub const fn handoff_read_fd(&self) -> u32 { self.handoff_read_fd }
    pub fn pipe_target(&self) -> &str { &self.pipe_target }
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
    pub fn checkpoint_one_claimed_observation_digest(&self) -> &str {
        &self.checkpoint_one_claimed_observation_digest
    }
    pub const fn previous_checkpoint_counter(&self) -> u64 { self.previous_checkpoint_counter }
    pub const fn checkpoint_two_counter(&self) -> u64 { self.checkpoint_two_counter }
    pub fn checkpoint_two_challenge_digest(&self) -> &str { &self.checkpoint_two_challenge_digest }
    pub fn checkpoint_two_challenge_nonce_blake3_hex(&self) -> &str {
        &self.checkpoint_two_challenge_nonce_blake3_hex
    }
    pub fn checkpoint_two_observation_qualification_digest(&self) -> &str {
        &self.checkpoint_two_observation_qualification_digest
    }
    pub fn checkpoint_two_observation_report_digest(&self) -> &str {
        &self.checkpoint_two_observation_report_digest
    }
    pub fn checkpoint_two_mapped_object_set_digest(&self) -> &str {
        &self.checkpoint_two_mapped_object_set_digest
    }
    pub const fn checkpoint_two_observed_at_ms(&self) -> u64 { self.checkpoint_two_observed_at_ms }

    pub const fn exact_supervisor_issued_ticket_bytes_consumed_by_tracee(&self) -> bool { true }
    pub const fn exact_successful_release_bound_to_tracee_observation(&self) -> bool { true }
    pub const fn release_precedes_challenge_by_capability_order(&self) -> bool { true }
    pub const fn challenge_consumption_precedes_live_observation(&self) -> bool { true }
    pub const fn causal_release_to_checkpoint_two_live_observation_established(&self) -> bool {
        true
    }
    pub const fn same_tracee_pid_and_runtime_process_claim_bound_across_handoff(&self) -> bool {
        true
    }
    pub const fn checkpoint_two_challenge_is_os_csprng_post_release_challenge(&self) -> bool {
        true
    }
    pub const fn checkpoint_one_live_observation_proven_here(&self) -> bool { false }
    pub const fn checkpoint_two_signed_inclusion_established(&self) -> bool { false }
    pub const fn mapping_continuity_between_checkpoints_established(&self) -> bool { false }
    pub const fn boot_measurement_live_reobserved_after_release(&self) -> bool { false }
    pub const fn runtime_config_live_reobserved_after_release(&self) -> bool { false }
    pub const fn exclusive_pipe_peer_authority_established(&self) -> bool { false }
    pub const fn trusted_time_established(&self) -> bool { false }
    pub const fn global_replay_excluded(&self) -> bool { false }
    pub const fn grants_physical_authority(&self) -> bool { false }
}

pub struct AuthenticatedPostReleaseCheckpointTwoQualification {
    pub report: AuthenticatedPostReleaseCheckpointTwoReport,
    authenticated: AuthenticatedPostReleaseCheckpointTwoObservation,
}

impl AuthenticatedPostReleaseCheckpointTwoQualification {
    pub fn authenticated(&self) -> &AuthenticatedPostReleaseCheckpointTwoObservation {
        &self.authenticated
    }
    pub fn into_authenticated(self) -> AuthenticatedPostReleaseCheckpointTwoObservation {
        self.authenticated
    }
}

pub fn bind_authenticated_post_release_checkpoint_two(
    policy: &AuthenticatedPostReleaseCheckpointTwoPolicy,
    issued: &IssuedPostReleaseRuntimeChallenge,
    observed: &PostReleaseCheckpointTwoObservation,
) -> Result<
    AuthenticatedPostReleaseCheckpointTwoQualification,
    AuthenticatedPostReleaseCheckpointTwoReport,
> {
    let policy_digest = policy.canonical_digest();
    let mut report = AuthenticatedPostReleaseCheckpointTwoReport {
        schema_version: AUTHENTICATED_POST_RELEASE_CHECKPOINT_TWO_REPORT_SCHEMA_V1.into(),
        policy_id: policy.policy_id.clone(),
        policy_digest: policy_digest.clone(),
        issuer_qualification_digest: issued.qualification_digest().into(),
        observation_qualification_digest: observed.qualification_digest().into(),
        ticket_digest: issued.ticket_digest().into(),
        ticket_bytes_digest: issued.ticket_bytes_digest().into(),
        channel_retention_digest: issued.channel_retention_digest().into(),
        release_digest: issued.release_digest().into(),
        release_report_digest: issued.release_report_digest().into(),
        confinement_qualification_digest: issued.confinement_qualification_digest().into(),
        mapping_continuity_qualification_digest: issued
            .mapping_continuity_qualification_digest()
            .into(),
        bootstrap_ready_wire_digest: issued.bootstrap_ready_wire_digest().into(),
        bootstrap_ready_checkpoint_digest: issued.bootstrap_ready_checkpoint_digest().into(),
        launch_handoff_qualification_digest: issued.launch_handoff_qualification_digest().into(),
        launch_ticket_digest: issued.launch_ticket_digest().into(),
        tracee_pid: issued.tracee_pid(),
        handoff_read_fd: issued.tracee_handoff_read_fd(),
        pipe_target: issued.pipe_target().into(),
        process_instance_id: issued.process_instance_id().into(),
        runtime_policy_digest: issued.runtime_policy_digest().into(),
        runtime_verifier_ref: issued.runtime_verifier_ref().into(),
        backend_id: issued.backend_id().into(),
        executable_digest: issued.executable_digest().into(),
        dependency_closure_digest: issued.dependency_closure_digest().into(),
        checkpoint_two_challenge_digest: observed.checkpoint_challenge_digest().into(),
        checkpoint_two_observation_qualification_digest: observed
            .raw_observation_qualification_digest()
            .into(),
        checkpoint_two_mapped_object_set_digest: observed.mapped_object_set_digest().into(),
        checkpoint_two_counter: observed.checkpoint_monotonic_counter(),
        previous_checkpoint_counter: observed.previous_checkpoint_monotonic_counter(),
        checkpoint_two_observed_at_ms: observed.observed_at_ms(),
        disposition: AuthenticatedPostReleaseCheckpointTwoDisposition::Invalid,
        issues: Vec::new(),
    };

    if !policy.validate() {
        report
            .issues
            .push(AuthenticatedPostReleaseCheckpointTwoIssue::InvalidPolicy);
        return Err(finalize(report));
    }
    if issued.policy_digest() != policy.expected_issuer_policy_digest {
        report
            .issues
            .push(AuthenticatedPostReleaseCheckpointTwoIssue::IssuerPolicyMismatch);
    }
    if observed.policy_digest() != policy.expected_tracee_observation_policy_digest {
        report
            .issues
            .push(AuthenticatedPostReleaseCheckpointTwoIssue::ObservationPolicyMismatch);
    }
    if issued.ticket_digest() != observed.ticket_digest() {
        report
            .issues
            .push(AuthenticatedPostReleaseCheckpointTwoIssue::TicketDigestMismatch);
    }
    if issued.ticket_bytes_digest() != observed.ticket_bytes_digest() {
        report
            .issues
            .push(AuthenticatedPostReleaseCheckpointTwoIssue::TicketBytesMismatch);
    }
    if issued.channel_retention_digest() != observed.channel_retention_digest() {
        report
            .issues
            .push(AuthenticatedPostReleaseCheckpointTwoIssue::RetentionMismatch);
    }
    if issued.release_digest() != observed.release_digest() {
        report
            .issues
            .push(AuthenticatedPostReleaseCheckpointTwoIssue::ReleaseMismatch);
    }
    if issued.release_report_digest() != observed.release_report_digest() {
        report
            .issues
            .push(AuthenticatedPostReleaseCheckpointTwoIssue::ReleaseReportMismatch);
    }
    if issued.confinement_qualification_digest() != observed.confinement_qualification_digest() {
        report
            .issues
            .push(AuthenticatedPostReleaseCheckpointTwoIssue::ConfinementMismatch);
    }
    if issued.mapping_continuity_qualification_digest()
        != observed.mapping_continuity_qualification_digest()
    {
        report
            .issues
            .push(AuthenticatedPostReleaseCheckpointTwoIssue::MappingContinuityMismatch);
    }
    if issued.bootstrap_ready_wire_digest() != observed.bootstrap_ready_wire_digest() {
        report
            .issues
            .push(AuthenticatedPostReleaseCheckpointTwoIssue::BootstrapReadyWireMismatch);
    }
    if issued.bootstrap_ready_checkpoint_digest() != observed.bootstrap_ready_checkpoint_digest() {
        report
            .issues
            .push(AuthenticatedPostReleaseCheckpointTwoIssue::BootstrapReadyCheckpointMismatch);
    }
    if issued.launch_handoff_qualification_digest() != observed.launch_handoff_qualification_digest()
    {
        report
            .issues
            .push(AuthenticatedPostReleaseCheckpointTwoIssue::LaunchHandoffMismatch);
    }
    if issued.launch_ticket_digest() != observed.launch_ticket_digest() {
        report
            .issues
            .push(AuthenticatedPostReleaseCheckpointTwoIssue::LaunchTicketMismatch);
    }
    if issued.tracee_pid() != observed.current_pid() {
        report
            .issues
            .push(AuthenticatedPostReleaseCheckpointTwoIssue::TraceePidMismatch);
    }
    if issued.tracee_handoff_read_fd() != observed.handoff_read_fd() {
        report
            .issues
            .push(AuthenticatedPostReleaseCheckpointTwoIssue::HandoffReadFdMismatch);
    }
    if issued.pipe_target() != observed.pipe_target() {
        report
            .issues
            .push(AuthenticatedPostReleaseCheckpointTwoIssue::PipeTargetMismatch);
    }
    if issued.process_instance_id() != observed.process_instance_id() {
        report
            .issues
            .push(AuthenticatedPostReleaseCheckpointTwoIssue::ProcessInstanceMismatch);
    }
    if issued.runtime_policy_digest() != policy.expected_runtime_policy_digest
        || observed.runtime_policy_digest() != policy.expected_runtime_policy_digest
        || issued.runtime_policy_digest() != observed.runtime_policy_digest()
    {
        report
            .issues
            .push(AuthenticatedPostReleaseCheckpointTwoIssue::RuntimePolicyMismatch);
    }
    if issued.runtime_verifier_ref() != policy.expected_runtime_verifier_ref
        || observed.runtime_verifier_ref() != policy.expected_runtime_verifier_ref
        || issued.runtime_verifier_ref() != observed.runtime_verifier_ref()
    {
        report
            .issues
            .push(AuthenticatedPostReleaseCheckpointTwoIssue::RuntimeVerifierMismatch);
    }
    if issued.backend_id() != policy.expected_backend_id
        || observed.backend_id() != policy.expected_backend_id
        || issued.backend_id() != observed.backend_id()
    {
        report
            .issues
            .push(AuthenticatedPostReleaseCheckpointTwoIssue::BackendMismatch);
    }
    if issued.boot_measurement_digest() != observed.boot_measurement_digest() {
        report
            .issues
            .push(AuthenticatedPostReleaseCheckpointTwoIssue::BootMeasurementMismatch);
    }
    if issued.executable_digest() != observed.executable_digest() {
        report
            .issues
            .push(AuthenticatedPostReleaseCheckpointTwoIssue::ExecutableMismatch);
    }
    if issued.dependency_closure_digest() != observed.dependency_closure_digest() {
        report
            .issues
            .push(AuthenticatedPostReleaseCheckpointTwoIssue::DependencyClosureMismatch);
    }
    if issued.runtime_config_digest() != observed.runtime_config_digest() {
        report
            .issues
            .push(AuthenticatedPostReleaseCheckpointTwoIssue::RuntimeConfigMismatch);
    }
    if issued.launch_attestation_digest() != observed.launch_attestation_digest() {
        report
            .issues
            .push(AuthenticatedPostReleaseCheckpointTwoIssue::LaunchAttestationMismatch);
    }
    if issued.checkpoint_one_challenge_digest() != observed.checkpoint_one_challenge_digest() {
        report
            .issues
            .push(AuthenticatedPostReleaseCheckpointTwoIssue::CheckpointOneChallengeMismatch);
    }
    if issued.checkpoint_one_observation_qualification_digest()
        != observed.checkpoint_one_claimed_observation_digest()
    {
        report.issues.push(
            AuthenticatedPostReleaseCheckpointTwoIssue::CheckpointOneClaimedObservationMismatch,
        );
    }
    if issued.checkpoint_sequence() != 2 || observed.checkpoint_sequence() != 2 {
        report
            .issues
            .push(AuthenticatedPostReleaseCheckpointTwoIssue::CheckpointSequenceMismatch);
    }
    if issued.previous_checkpoint_digest() != observed.previous_checkpoint_digest()
        || issued.previous_checkpoint_digest() != issued.bootstrap_ready_checkpoint_digest()
        || observed.previous_checkpoint_digest() != observed.bootstrap_ready_checkpoint_digest()
    {
        report
            .issues
            .push(AuthenticatedPostReleaseCheckpointTwoIssue::PreviousCheckpointMismatch);
    }
    if issued.previous_checkpoint_monotonic_counter()
        != observed.previous_checkpoint_monotonic_counter()
    {
        report
            .issues
            .push(AuthenticatedPostReleaseCheckpointTwoIssue::PreviousCounterMismatch);
    }
    if issued.challenge_nonce_blake3_hex() != observed.challenge_nonce_blake3_hex() {
        report
            .issues
            .push(AuthenticatedPostReleaseCheckpointTwoIssue::ChallengeNonceMismatch);
    }

    if !report.issues.is_empty() {
        return Err(finalize(report));
    }

    report.disposition = AuthenticatedPostReleaseCheckpointTwoDisposition::Qualified;
    let report_digest = report.canonical_digest();
    let policy_digest = policy_digest.expect("validated policy has digest");
    let qualification_digest = qualification_digest(
        &policy_digest,
        issued.qualification_digest(),
        observed.qualification_digest(),
        issued.ticket_digest(),
        issued.ticket_bytes_digest(),
        issued.release_digest(),
        observed.checkpoint_challenge_digest(),
        observed.raw_observation_qualification_digest(),
        &report_digest,
    );
    let authenticated = AuthenticatedPostReleaseCheckpointTwoObservation {
        qualification_digest,
        report_digest,
        policy_digest,
        issuer_qualification_digest: issued.qualification_digest().into(),
        tracee_observation_qualification_digest: observed.qualification_digest().into(),
        ticket_digest: issued.ticket_digest().into(),
        ticket_bytes_digest: issued.ticket_bytes_digest().into(),
        channel_retention_digest: issued.channel_retention_digest().into(),
        release_digest: issued.release_digest().into(),
        release_report_digest: issued.release_report_digest().into(),
        confinement_qualification_digest: issued.confinement_qualification_digest().into(),
        mapping_continuity_qualification_digest: issued
            .mapping_continuity_qualification_digest()
            .into(),
        bootstrap_ready_wire_digest: issued.bootstrap_ready_wire_digest().into(),
        bootstrap_ready_checkpoint_digest: issued.bootstrap_ready_checkpoint_digest().into(),
        launch_handoff_qualification_digest: issued.launch_handoff_qualification_digest().into(),
        launch_ticket_digest: issued.launch_ticket_digest().into(),
        tracee_pid: issued.tracee_pid(),
        handoff_read_fd: issued.tracee_handoff_read_fd(),
        pipe_target: issued.pipe_target().into(),
        process_instance_id: issued.process_instance_id().into(),
        runtime_policy_digest: issued.runtime_policy_digest().into(),
        runtime_verifier_ref: issued.runtime_verifier_ref().into(),
        backend_id: issued.backend_id().into(),
        boot_measurement_digest: issued.boot_measurement_digest().into(),
        executable_digest: issued.executable_digest().into(),
        dependency_closure_digest: issued.dependency_closure_digest().into(),
        runtime_config_digest: issued.runtime_config_digest().into(),
        launch_attestation_digest: issued.launch_attestation_digest().into(),
        checkpoint_one_challenge_digest: issued.checkpoint_one_challenge_digest().into(),
        checkpoint_one_claimed_observation_digest: issued
            .checkpoint_one_observation_qualification_digest()
            .into(),
        previous_checkpoint_counter: issued.previous_checkpoint_monotonic_counter(),
        checkpoint_two_counter: observed.checkpoint_monotonic_counter(),
        checkpoint_two_challenge_digest: observed.checkpoint_challenge_digest().into(),
        checkpoint_two_challenge_nonce_blake3_hex: issued.challenge_nonce_blake3_hex().into(),
        checkpoint_two_observation_qualification_digest: observed
            .raw_observation_qualification_digest()
            .into(),
        checkpoint_two_observation_report_digest: observed
            .raw_observation_report_digest()
            .into(),
        checkpoint_two_mapped_object_set_digest: observed.mapped_object_set_digest().into(),
        checkpoint_two_observed_at_ms: observed.observed_at_ms(),
    };

    Ok(AuthenticatedPostReleaseCheckpointTwoQualification {
        report,
        authenticated,
    })
}

#[allow(clippy::too_many_arguments)]
fn qualification_digest(
    policy: &str,
    issuer: &str,
    observation: &str,
    ticket: &str,
    ticket_bytes: &str,
    release: &str,
    challenge: &str,
    live_observation: &str,
    report: &str,
) -> String {
    let mut h = blake3::Hasher::new();
    h.update(QUALIFICATION_DOMAIN);
    for value in [
        policy,
        issuer,
        observation,
        ticket,
        ticket_bytes,
        release,
        challenge,
        live_observation,
        report,
    ] {
        field(&mut h, value);
    }
    b3(h.finalize())
}

fn finalize(
    mut report: AuthenticatedPostReleaseCheckpointTwoReport,
) -> AuthenticatedPostReleaseCheckpointTwoReport {
    report.disposition = if report
        .issues
        .iter()
        .any(AuthenticatedPostReleaseCheckpointTwoIssue::invalid)
    {
        AuthenticatedPostReleaseCheckpointTwoDisposition::Invalid
    } else {
        AuthenticatedPostReleaseCheckpointTwoDisposition::Blocked
    };
    report
}

fn valid_blake3(value: &str) -> bool {
    value.strip_prefix("blake3:").is_some_and(|digest| {
        digest.len() == 64
            && digest
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    })
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

fn b3(hash: blake3::Hash) -> String {
    format!("blake3:{}", hash.to_hex())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(label: &str) -> String {
        format!("blake3:{}", blake3::hash(label.as_bytes()).to_hex())
    }

    fn policy() -> AuthenticatedPostReleaseCheckpointTwoPolicy {
        AuthenticatedPostReleaseCheckpointTwoPolicy {
            schema_version: AUTHENTICATED_POST_RELEASE_CHECKPOINT_TWO_POLICY_SCHEMA_V1.into(),
            policy_id: "authenticated-post-release-checkpoint-two:v1".into(),
            expected_issuer_policy_digest: d("issuer-policy"),
            expected_tracee_observation_policy_digest: d("tracee-policy"),
            expected_runtime_policy_digest: d("runtime-policy"),
            expected_runtime_verifier_ref: "verifier:prod".into(),
            expected_backend_id: "backend:in-process-p256".into(),
            evidence_refs: vec!["review:authenticated-post-release-checkpoint-two".into()],
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
    fn both_parent_policies_and_runtime_role_are_semantic() {
        let left = policy();
        let mut right = left.clone();
        right.expected_issuer_policy_digest = d("different-issuer");
        assert_ne!(left.canonical_digest(), right.canonical_digest());
        let mut right = left.clone();
        right.expected_tracee_observation_policy_digest = d("different-tracee");
        assert_ne!(left.canonical_digest(), right.canonical_digest());
        let mut right = left.clone();
        right.expected_runtime_verifier_ref = "verifier:other".into();
        assert_ne!(left.canonical_digest(), right.canonical_digest());
    }

    #[test]
    fn final_identity_binds_both_parents_ticket_release_and_live_observation() {
        let base = qualification_digest(
            &d("policy"),
            &d("issuer"),
            &d("tracee"),
            &d("ticket"),
            &d("ticket-bytes"),
            &d("release"),
            &d("challenge"),
            &d("live-observation"),
            &d("report"),
        );
        let changed = qualification_digest(
            &d("policy"),
            &d("issuer"),
            &d("tracee"),
            &d("ticket"),
            &d("ticket-bytes"),
            &d("release"),
            &d("challenge"),
            &d("different-live-observation"),
            &d("report"),
        );
        assert_ne!(base, changed);
    }

    #[test]
    fn claim_ceiling_stays_explicit() {
        let claims = [
            "checkpoint-one-live-proven=false",
            "checkpoint-two-signed-inclusion=false",
            "between-checkpoint-continuity=false",
            "boot-live-reobserved=false",
            "runtime-config-live-reobserved=false",
            "exclusive-pipe-peer=false",
            "trusted-time=false",
            "global-replay=false",
            "physical-authority=false",
        ];
        assert_eq!(claims.len(), 9);
    }
}
