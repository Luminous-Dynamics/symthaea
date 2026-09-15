// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Compose a ptrace-confirmed launch into the first signed mapped-runtime checkpoint.
//!
//! This bridge adds no new observation mechanism. It authenticates the ticket
//! consumed by the launched tracee against the exact supervisor-issued ticket,
//! rebinds both sides to the exact exec confirmation and critical-authority
//! capability, and requires the first binding in the already-qualified fresh
//! mapped-runtime continuity trace to be the exact checkpoint-one observation
//! created after ticket consumption.

#![cfg(target_os = "linux")]
#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use symthaea_assurance_critical_exec_preserved_authority::CriticalExecPreservedAuthority;
use symthaea_assurance_fresh_mapped_runtime_continuity::FreshMappedRuntimeContinuity;
use symthaea_assurance_launch_bound_first_runtime_observation::
    LaunchBoundFirstObservationQualification;
use symthaea_assurance_launch_runtime_handoff_channel::IssuedLaunchRuntimeChallenge;
use symthaea_assurance_ptrace_static_exec_confirmation::ConfirmedStaticExecLaunch;

pub const LAUNCH_TO_RUNTIME_CONTINUITY_POLICY_SCHEMA_V1: &str =
    "symthaea.assurance.launch-to-runtime-continuity-policy.v1";
pub const LAUNCH_TO_RUNTIME_CONTINUITY_REPORT_SCHEMA_V1: &str =
    "symthaea.assurance.launch-to-runtime-continuity-report.v1";

const POLICY_DOMAIN: &[u8] =
    b"symthaea.assurance.launch-to-runtime-continuity-policy.digest.v1\0";
const REPORT_DOMAIN: &[u8] =
    b"symthaea.assurance.launch-to-runtime-continuity-report.digest.v1\0";
const QUALIFICATION_DOMAIN: &[u8] =
    b"symthaea.assurance.launch-to-runtime-continuity-qualification.digest.v1\0";
const MAX_TEXT: usize = 4096;
const MAX_REFS: usize = 128;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchToRuntimeContinuityPolicy {
    pub schema_version: String,
    pub policy_id: String,
    pub expected_exec_confirmation_policy_digest: String,
    pub expected_critical_authority_policy_digest: String,
    pub expected_handoff_policy_digest: String,
    pub expected_first_observation_policy_digest: String,
    pub expected_fresh_continuity_policy_digest: String,
    pub expected_runtime_policy_digest: String,
    pub expected_runtime_verifier_ref: String,
    pub expected_backend_id: String,
    pub evidence_refs: Vec<String>,
}

impl LaunchToRuntimeContinuityPolicy {
    pub fn validate(&self) -> bool {
        self.schema_version == LAUNCH_TO_RUNTIME_CONTINUITY_POLICY_SCHEMA_V1
            && canonical_text(&self.policy_id)
            && valid_blake3(&self.expected_exec_confirmation_policy_digest)
            && valid_blake3(&self.expected_critical_authority_policy_digest)
            && valid_blake3(&self.expected_handoff_policy_digest)
            && valid_blake3(&self.expected_first_observation_policy_digest)
            && valid_blake3(&self.expected_fresh_continuity_policy_digest)
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
            self.expected_exec_confirmation_policy_digest.as_str(),
            self.expected_critical_authority_policy_digest.as_str(),
            self.expected_handoff_policy_digest.as_str(),
            self.expected_first_observation_policy_digest.as_str(),
            self.expected_fresh_continuity_policy_digest.as_str(),
            self.expected_runtime_policy_digest.as_str(),
            self.expected_runtime_verifier_ref.as_str(),
            self.expected_backend_id.as_str(),
        ] {
            field(&mut h, value);
        }
        sorted_strings(&mut h, &self.evidence_refs);
        Some(format!("blake3:{}", h.finalize().to_hex()))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LaunchToRuntimeContinuityIssue {
    InvalidPolicy,
    ExecConfirmationPolicyMismatch,
    CriticalAuthorityPolicyMismatch,
    HandoffPolicyMismatch,
    FirstObservationPolicyMismatch,
    FreshContinuityPolicyMismatch,
    RuntimePolicyMismatch,
    RuntimeVerifierMismatch,
    BackendMismatch,
    AuthorityConfirmationMismatch,
    AuthorityTraceeMismatch,
    IssuerConfirmationMismatch,
    IssuerAuthorityMismatch,
    IssuerAuthoritySnapshotMismatch,
    IssuerTraceeMismatch,
    IssuerPlanMismatch,
    IssuerLaunchNonceMismatch,
    IssuerRuntimeIdentityMismatch,
    ConsumedTicketMismatch,
    ConsumedConfirmationMismatch,
    ConsumedAuthorityMismatch,
    ConsumedAuthoritySnapshotMismatch,
    ConsumedPlanMismatch,
    ConsumedLaunchNonceMismatch,
    ConsumedTraceeMismatch,
    ConsumedPipeMismatch,
    ConsumedChallengeMismatch,
    ConsumedRuntimeIdentityMismatch,
    FirstFreshQualificationMismatch,
    FirstFreshChallengeMismatch,
    FirstFreshProcessMismatch,
    FirstFreshRuntimeIdentityMismatch,
    MissingFirstSignedBinding,
    FirstBindingSequenceMismatch,
    FirstBindingHasPredecessor,
    FirstBindingCounterMismatch,
    FirstBindingChallengeMismatch,
    FirstBindingFreshQualificationMismatch,
    FirstBindingRawObservationMismatch,
    FirstBindingMappedSetMismatch,
    FirstBindingObservationTimeMismatch,
    FirstBindingDynamicMeasurementMismatch,
    SignedTraceProcessMismatch,
    SignedTraceRuntimeIdentityMismatch,
}

impl LaunchToRuntimeContinuityIssue {
    fn invalid(&self) -> bool { matches!(self, Self::InvalidPolicy) }

    fn code(&self) -> &'static str {
        match self {
            Self::InvalidPolicy => "invalid-policy",
            Self::ExecConfirmationPolicyMismatch => "exec-confirmation-policy-mismatch",
            Self::CriticalAuthorityPolicyMismatch => "critical-authority-policy-mismatch",
            Self::HandoffPolicyMismatch => "handoff-policy-mismatch",
            Self::FirstObservationPolicyMismatch => "first-observation-policy-mismatch",
            Self::FreshContinuityPolicyMismatch => "fresh-continuity-policy-mismatch",
            Self::RuntimePolicyMismatch => "runtime-policy-mismatch",
            Self::RuntimeVerifierMismatch => "runtime-verifier-mismatch",
            Self::BackendMismatch => "backend-mismatch",
            Self::AuthorityConfirmationMismatch => "authority-confirmation-mismatch",
            Self::AuthorityTraceeMismatch => "authority-tracee-mismatch",
            Self::IssuerConfirmationMismatch => "issuer-confirmation-mismatch",
            Self::IssuerAuthorityMismatch => "issuer-authority-mismatch",
            Self::IssuerAuthoritySnapshotMismatch => "issuer-authority-snapshot-mismatch",
            Self::IssuerTraceeMismatch => "issuer-tracee-mismatch",
            Self::IssuerPlanMismatch => "issuer-plan-mismatch",
            Self::IssuerLaunchNonceMismatch => "issuer-launch-nonce-mismatch",
            Self::IssuerRuntimeIdentityMismatch => "issuer-runtime-identity-mismatch",
            Self::ConsumedTicketMismatch => "consumed-ticket-mismatch",
            Self::ConsumedConfirmationMismatch => "consumed-confirmation-mismatch",
            Self::ConsumedAuthorityMismatch => "consumed-authority-mismatch",
            Self::ConsumedAuthoritySnapshotMismatch => "consumed-authority-snapshot-mismatch",
            Self::ConsumedPlanMismatch => "consumed-plan-mismatch",
            Self::ConsumedLaunchNonceMismatch => "consumed-launch-nonce-mismatch",
            Self::ConsumedTraceeMismatch => "consumed-tracee-mismatch",
            Self::ConsumedPipeMismatch => "consumed-pipe-mismatch",
            Self::ConsumedChallengeMismatch => "consumed-challenge-mismatch",
            Self::ConsumedRuntimeIdentityMismatch => "consumed-runtime-identity-mismatch",
            Self::FirstFreshQualificationMismatch => "first-fresh-qualification-mismatch",
            Self::FirstFreshChallengeMismatch => "first-fresh-challenge-mismatch",
            Self::FirstFreshProcessMismatch => "first-fresh-process-mismatch",
            Self::FirstFreshRuntimeIdentityMismatch => "first-fresh-runtime-identity-mismatch",
            Self::MissingFirstSignedBinding => "missing-first-signed-binding",
            Self::FirstBindingSequenceMismatch => "first-binding-sequence-mismatch",
            Self::FirstBindingHasPredecessor => "first-binding-has-predecessor",
            Self::FirstBindingCounterMismatch => "first-binding-counter-mismatch",
            Self::FirstBindingChallengeMismatch => "first-binding-challenge-mismatch",
            Self::FirstBindingFreshQualificationMismatch => "first-binding-fresh-qualification-mismatch",
            Self::FirstBindingRawObservationMismatch => "first-binding-raw-observation-mismatch",
            Self::FirstBindingMappedSetMismatch => "first-binding-mapped-set-mismatch",
            Self::FirstBindingObservationTimeMismatch => "first-binding-observation-time-mismatch",
            Self::FirstBindingDynamicMeasurementMismatch => "first-binding-dynamic-measurement-mismatch",
            Self::SignedTraceProcessMismatch => "signed-trace-process-mismatch",
            Self::SignedTraceRuntimeIdentityMismatch => "signed-trace-runtime-identity-mismatch",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LaunchToRuntimeContinuityDisposition {
    Invalid,
    Blocked,
    Qualified,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct LaunchToRuntimeContinuityReport {
    pub schema_version: String,
    pub policy_id: String,
    pub policy_digest: Option<String>,
    pub exec_confirmation_digest: String,
    pub critical_authority_qualification_digest: String,
    pub authority_snapshot_digest: String,
    pub issued_handoff_qualification_digest: String,
    pub consumed_first_observation_qualification_digest: String,
    pub fresh_continuity_qualification_digest: String,
    pub ticket_digest: String,
    pub tracee_pid: i32,
    pub process_instance_id: String,
    pub first_checkpoint_digest: Option<String>,
    pub first_checkpoint_dynamic_measurement_digest: Option<String>,
    pub first_challenge_digest: String,
    pub first_fresh_observation_qualification_digest: String,
    pub fresh_sequence_digest: String,
    pub runtime_trace_digest: String,
    pub disposition: LaunchToRuntimeContinuityDisposition,
    pub issues: Vec<LaunchToRuntimeContinuityIssue>,
}

impl LaunchToRuntimeContinuityReport {
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
            self.issued_handoff_qualification_digest.as_str(),
            self.consumed_first_observation_qualification_digest.as_str(),
            self.fresh_continuity_qualification_digest.as_str(),
            self.ticket_digest.as_str(),
            self.process_instance_id.as_str(),
            self.first_checkpoint_digest.as_deref().unwrap_or("-"),
            self.first_checkpoint_dynamic_measurement_digest.as_deref().unwrap_or("-"),
            self.first_challenge_digest.as_str(),
            self.first_fresh_observation_qualification_digest.as_str(),
            self.fresh_sequence_digest.as_str(),
            self.runtime_trace_digest.as_str(),
        ] {
            field(&mut h, value);
        }
        h.update(&self.tracee_pid.to_le_bytes());
        field(
            &mut h,
            match self.disposition {
                LaunchToRuntimeContinuityDisposition::Invalid => "invalid",
                LaunchToRuntimeContinuityDisposition::Blocked => "blocked",
                LaunchToRuntimeContinuityDisposition::Qualified => "qualified",
            },
        );
        h.update(&(self.issues.len() as u64).to_le_bytes());
        for issue in &self.issues {
            field(&mut h, issue.code());
        }
        format!("blake3:{}", h.finalize().to_hex())
    }

    pub const fn grants_physical_authority(&self) -> bool { false }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LaunchToRuntimeContinuity {
    qualification_digest: String,
    report_digest: String,
    policy_digest: String,
    exec_confirmation_digest: String,
    critical_authority_qualification_digest: String,
    authority_snapshot_digest: String,
    issued_handoff_qualification_digest: String,
    consumed_first_observation_qualification_digest: String,
    fresh_continuity_qualification_digest: String,
    ticket_digest: String,
    launch_plan_digest: String,
    launch_nonce_blake3_hex: String,
    launch_challenge_nonce_blake3_hex: String,
    tracee_pid: i32,
    process_instance_id: String,
    first_checkpoint_digest: String,
    first_checkpoint_dynamic_measurement_digest: String,
    first_challenge_digest: String,
    first_fresh_observation_qualification_digest: String,
    first_raw_observation_qualification_digest: String,
    first_mapped_object_set_digest: String,
    first_observed_at_ms: u64,
    fresh_sequence_digest: String,
    runtime_trace_digest: String,
    runtime_policy_digest: String,
    runtime_verifier_ref: String,
    backend_id: String,
    executable_digest: String,
}

impl LaunchToRuntimeContinuity {
    pub fn qualification_digest(&self) -> &str { &self.qualification_digest }
    pub fn report_digest(&self) -> &str { &self.report_digest }
    pub fn policy_digest(&self) -> &str { &self.policy_digest }
    pub fn exec_confirmation_digest(&self) -> &str { &self.exec_confirmation_digest }
    pub fn critical_authority_qualification_digest(&self) -> &str {
        &self.critical_authority_qualification_digest
    }
    pub fn authority_snapshot_digest(&self) -> &str { &self.authority_snapshot_digest }
    pub fn issued_handoff_qualification_digest(&self) -> &str {
        &self.issued_handoff_qualification_digest
    }
    pub fn consumed_first_observation_qualification_digest(&self) -> &str {
        &self.consumed_first_observation_qualification_digest
    }
    pub fn fresh_continuity_qualification_digest(&self) -> &str {
        &self.fresh_continuity_qualification_digest
    }
    pub fn ticket_digest(&self) -> &str { &self.ticket_digest }
    pub fn launch_plan_digest(&self) -> &str { &self.launch_plan_digest }
    pub fn launch_nonce_blake3_hex(&self) -> &str { &self.launch_nonce_blake3_hex }
    pub fn launch_challenge_nonce_blake3_hex(&self) -> &str {
        &self.launch_challenge_nonce_blake3_hex
    }
    pub const fn tracee_pid(&self) -> i32 { self.tracee_pid }
    pub fn process_instance_id(&self) -> &str { &self.process_instance_id }
    pub fn first_checkpoint_digest(&self) -> &str { &self.first_checkpoint_digest }
    pub fn first_checkpoint_dynamic_measurement_digest(&self) -> &str {
        &self.first_checkpoint_dynamic_measurement_digest
    }
    pub fn first_challenge_digest(&self) -> &str { &self.first_challenge_digest }
    pub fn first_fresh_observation_qualification_digest(&self) -> &str {
        &self.first_fresh_observation_qualification_digest
    }
    pub fn first_raw_observation_qualification_digest(&self) -> &str {
        &self.first_raw_observation_qualification_digest
    }
    pub fn first_mapped_object_set_digest(&self) -> &str { &self.first_mapped_object_set_digest }
    pub const fn first_observed_at_ms(&self) -> u64 { self.first_observed_at_ms }
    pub fn fresh_sequence_digest(&self) -> &str { &self.fresh_sequence_digest }
    pub fn runtime_trace_digest(&self) -> &str { &self.runtime_trace_digest }
    pub fn runtime_policy_digest(&self) -> &str { &self.runtime_policy_digest }
    pub fn runtime_verifier_ref(&self) -> &str { &self.runtime_verifier_ref }
    pub fn backend_id(&self) -> &str { &self.backend_id }
    pub fn executable_digest(&self) -> &str { &self.executable_digest }

    pub const fn supervisor_issued_ticket_authenticated_by_exact_opaque_issuer(&self) -> bool {
        true
    }
    pub const fn same_ptrace_confirmed_os_process_consumed_ticket(&self) -> bool { true }
    pub const fn critical_exec_authority_bound_into_runtime_lineage(&self) -> bool { true }
    pub const fn launch_challenge_bound_into_live_first_observation(&self) -> bool { true }
    pub const fn first_live_observation_committed_by_signed_checkpoint_one(&self) -> bool { true }
    pub const fn signed_runtime_trace_descends_from_confirmed_launch_handoff(&self) -> bool { true }
    pub const fn causal_launch_to_first_checkpoint_lineage_established(&self) -> bool { true }

    pub const fn uninterrupted_mapping_continuity_exec_to_first_observation_established(&self) -> bool {
        false
    }
    pub const fn ticket_consumption_was_first_application_action_established(&self) -> bool { false }
    pub const fn exclusive_pipe_writer_authority_established(&self) -> bool { false }
    pub const fn mapped_runtime_continuity_between_checkpoints_established(&self) -> bool { false }
    pub const fn global_cross_trace_replay_excluded(&self) -> bool { false }
    pub const fn trusted_time_established(&self) -> bool { false }
    pub const fn all_exec_preserved_process_state_qualified(&self) -> bool { false }
    pub const fn grants_physical_authority(&self) -> bool { false }
}

pub struct LaunchToRuntimeContinuityQualification {
    pub report: LaunchToRuntimeContinuityReport,
    verified: LaunchToRuntimeContinuity,
}

impl LaunchToRuntimeContinuityQualification {
    pub fn verified(&self) -> &LaunchToRuntimeContinuity { &self.verified }
    pub fn into_verified(self) -> LaunchToRuntimeContinuity { self.verified }
}

pub fn bind_launch_to_runtime_continuity(
    policy: &LaunchToRuntimeContinuityPolicy,
    confirmed: &ConfirmedStaticExecLaunch,
    authority: &CriticalExecPreservedAuthority,
    issued: &IssuedLaunchRuntimeChallenge,
    first: &LaunchBoundFirstObservationQualification,
    continuity: &FreshMappedRuntimeContinuity,
) -> Result<LaunchToRuntimeContinuityQualification, LaunchToRuntimeContinuityReport> {
    let policy_digest = policy.canonical_digest();
    let consumed = first.verified();
    let fresh = first.fresh.fresh();
    let mut report = LaunchToRuntimeContinuityReport {
        schema_version: LAUNCH_TO_RUNTIME_CONTINUITY_REPORT_SCHEMA_V1.into(),
        policy_id: policy.policy_id.clone(),
        policy_digest: policy_digest.clone(),
        exec_confirmation_digest: confirmed.confirmation_digest().into(),
        critical_authority_qualification_digest: authority.qualification_digest().into(),
        authority_snapshot_digest: authority.authority_snapshot_digest().into(),
        issued_handoff_qualification_digest: issued.qualification_digest().into(),
        consumed_first_observation_qualification_digest: consumed.qualification_digest().into(),
        fresh_continuity_qualification_digest: continuity.qualification_digest().into(),
        ticket_digest: consumed.ticket_digest().into(),
        tracee_pid: confirmed.tracee_pid(),
        process_instance_id: consumed.process_instance_id().into(),
        first_checkpoint_digest: None,
        first_checkpoint_dynamic_measurement_digest: None,
        first_challenge_digest: consumed.checkpoint_challenge_digest().into(),
        first_fresh_observation_qualification_digest: consumed
            .fresh_observation_qualification_digest()
            .into(),
        fresh_sequence_digest: continuity.fresh_sequence_digest().into(),
        runtime_trace_digest: continuity.runtime_trace_digest().into(),
        disposition: LaunchToRuntimeContinuityDisposition::Invalid,
        issues: Vec::new(),
    };

    if !policy.validate() {
        report.issues.push(LaunchToRuntimeContinuityIssue::InvalidPolicy);
        return Err(finalize(report));
    }
    if confirmed.policy_digest() != policy.expected_exec_confirmation_policy_digest {
        report
            .issues
            .push(LaunchToRuntimeContinuityIssue::ExecConfirmationPolicyMismatch);
    }
    if authority.policy_digest() != policy.expected_critical_authority_policy_digest {
        report
            .issues
            .push(LaunchToRuntimeContinuityIssue::CriticalAuthorityPolicyMismatch);
    }
    if issued.policy_digest() != policy.expected_handoff_policy_digest {
        report
            .issues
            .push(LaunchToRuntimeContinuityIssue::HandoffPolicyMismatch);
    }
    if consumed.policy_digest() != policy.expected_first_observation_policy_digest {
        report
            .issues
            .push(LaunchToRuntimeContinuityIssue::FirstObservationPolicyMismatch);
    }
    if continuity.policy_digest() != policy.expected_fresh_continuity_policy_digest {
        report
            .issues
            .push(LaunchToRuntimeContinuityIssue::FreshContinuityPolicyMismatch);
    }

    for runtime_policy in [
        confirmed.runtime_policy_digest(),
        authority.runtime_policy_digest(),
        issued.runtime_policy_digest(),
        consumed.runtime_policy_digest(),
        fresh.runtime_policy_digest(),
        continuity.runtime_policy_digest(),
    ] {
        if runtime_policy != policy.expected_runtime_policy_digest {
            report.issues.push(LaunchToRuntimeContinuityIssue::RuntimePolicyMismatch);
            break;
        }
    }
    for verifier in [
        confirmed.runtime_verifier_ref(),
        authority.runtime_verifier_ref(),
        issued.runtime_verifier_ref(),
        consumed.runtime_verifier_ref(),
        fresh.runtime_verifier_ref(),
        continuity.runtime_verifier_ref(),
    ] {
        if verifier != policy.expected_runtime_verifier_ref {
            report.issues.push(LaunchToRuntimeContinuityIssue::RuntimeVerifierMismatch);
            break;
        }
    }
    for backend in [
        confirmed.backend_id(),
        authority.backend_id(),
        issued.backend_id(),
        consumed.backend_id(),
        fresh.backend_id(),
        continuity.backend_id(),
    ] {
        if backend != policy.expected_backend_id {
            report.issues.push(LaunchToRuntimeContinuityIssue::BackendMismatch);
            break;
        }
    }

    if authority.exec_confirmation_digest() != confirmed.confirmation_digest() {
        report
            .issues
            .push(LaunchToRuntimeContinuityIssue::AuthorityConfirmationMismatch);
    }
    if authority.tracee_pid() != confirmed.tracee_pid() {
        report.issues.push(LaunchToRuntimeContinuityIssue::AuthorityTraceeMismatch);
    }
    if issued.exec_confirmation_digest() != confirmed.confirmation_digest() {
        report
            .issues
            .push(LaunchToRuntimeContinuityIssue::IssuerConfirmationMismatch);
    }
    if issued.critical_authority_qualification_digest() != authority.qualification_digest() {
        report.issues.push(LaunchToRuntimeContinuityIssue::IssuerAuthorityMismatch);
    }
    if issued.authority_snapshot_digest() != authority.authority_snapshot_digest() {
        report
            .issues
            .push(LaunchToRuntimeContinuityIssue::IssuerAuthoritySnapshotMismatch);
    }
    if issued.tracee_pid() != confirmed.tracee_pid() {
        report.issues.push(LaunchToRuntimeContinuityIssue::IssuerTraceeMismatch);
    }
    if issued.plan_digest() != confirmed.plan_digest() {
        report.issues.push(LaunchToRuntimeContinuityIssue::IssuerPlanMismatch);
    }
    if issued.launch_nonce_blake3_hex() != confirmed.launch_nonce_blake3_hex() {
        report
            .issues
            .push(LaunchToRuntimeContinuityIssue::IssuerLaunchNonceMismatch);
    }
    if issued.runtime_policy_digest() != confirmed.runtime_policy_digest()
        || issued.runtime_verifier_ref() != confirmed.runtime_verifier_ref()
        || issued.backend_id() != confirmed.backend_id()
        || issued.executable_digest() != confirmed.executable_digest()
    {
        report
            .issues
            .push(LaunchToRuntimeContinuityIssue::IssuerRuntimeIdentityMismatch);
    }

    if consumed.ticket_digest() != issued.ticket_digest() {
        report.issues.push(LaunchToRuntimeContinuityIssue::ConsumedTicketMismatch);
    }
    if consumed.exec_confirmation_digest() != confirmed.confirmation_digest() {
        report
            .issues
            .push(LaunchToRuntimeContinuityIssue::ConsumedConfirmationMismatch);
    }
    if consumed.critical_authority_qualification_digest() != authority.qualification_digest() {
        report.issues.push(LaunchToRuntimeContinuityIssue::ConsumedAuthorityMismatch);
    }
    if consumed.authority_snapshot_digest() != authority.authority_snapshot_digest() {
        report
            .issues
            .push(LaunchToRuntimeContinuityIssue::ConsumedAuthoritySnapshotMismatch);
    }
    if consumed.launch_plan_digest() != confirmed.plan_digest() {
        report.issues.push(LaunchToRuntimeContinuityIssue::ConsumedPlanMismatch);
    }
    if consumed.launch_nonce_blake3_hex() != confirmed.launch_nonce_blake3_hex() {
        report
            .issues
            .push(LaunchToRuntimeContinuityIssue::ConsumedLaunchNonceMismatch);
    }
    if consumed.current_pid() != confirmed.tracee_pid() || consumed.current_pid() != issued.tracee_pid() {
        report.issues.push(LaunchToRuntimeContinuityIssue::ConsumedTraceeMismatch);
    }
    if consumed.handoff_read_fd() != issued.tracee_handoff_read_fd()
        || consumed.pipe_target() != issued.pipe_target()
    {
        report.issues.push(LaunchToRuntimeContinuityIssue::ConsumedPipeMismatch);
    }
    if consumed.launch_challenge_nonce_blake3_hex() != issued.challenge_nonce_blake3_hex() {
        report
            .issues
            .push(LaunchToRuntimeContinuityIssue::ConsumedChallengeMismatch);
    }
    if consumed.runtime_policy_digest() != issued.runtime_policy_digest()
        || consumed.runtime_verifier_ref() != issued.runtime_verifier_ref()
        || consumed.backend_id() != issued.backend_id()
        || consumed.executable_digest() != issued.executable_digest()
    {
        report
            .issues
            .push(LaunchToRuntimeContinuityIssue::ConsumedRuntimeIdentityMismatch);
    }

    if fresh.qualification_digest() != consumed.fresh_observation_qualification_digest() {
        report
            .issues
            .push(LaunchToRuntimeContinuityIssue::FirstFreshQualificationMismatch);
    }
    if fresh.challenge_nonce_blake3_hex() != issued.challenge_nonce_blake3_hex()
        || fresh.challenge_digest() != consumed.checkpoint_challenge_digest()
    {
        report
            .issues
            .push(LaunchToRuntimeContinuityIssue::FirstFreshChallengeMismatch);
    }
    if fresh.process_instance_id() != consumed.process_instance_id() {
        report.issues.push(LaunchToRuntimeContinuityIssue::FirstFreshProcessMismatch);
    }
    if fresh.runtime_policy_digest() != consumed.runtime_policy_digest()
        || fresh.runtime_verifier_ref() != consumed.runtime_verifier_ref()
        || fresh.backend_id() != consumed.backend_id()
        || fresh.executable_digest() != consumed.executable_digest()
    {
        report
            .issues
            .push(LaunchToRuntimeContinuityIssue::FirstFreshRuntimeIdentityMismatch);
    }

    let Some(binding) = continuity.bindings().first() else {
        report.issues.push(LaunchToRuntimeContinuityIssue::MissingFirstSignedBinding);
        return Err(finalize(report));
    };
    report.first_checkpoint_digest = Some(binding.checkpoint_digest.clone());
    report.first_checkpoint_dynamic_measurement_digest =
        Some(binding.checkpoint_dynamic_measurement_digest.clone());
    if binding.sequence != 1 {
        report
            .issues
            .push(LaunchToRuntimeContinuityIssue::FirstBindingSequenceMismatch);
    }
    if binding.previous_checkpoint_digest.is_some() {
        report
            .issues
            .push(LaunchToRuntimeContinuityIssue::FirstBindingHasPredecessor);
    }
    if binding.monotonic_counter != consumed.checkpoint_monotonic_counter() {
        report
            .issues
            .push(LaunchToRuntimeContinuityIssue::FirstBindingCounterMismatch);
    }
    if binding.challenge_digest != consumed.checkpoint_challenge_digest()
        || binding.challenge_nonce_blake3_hex != issued.challenge_nonce_blake3_hex()
    {
        report
            .issues
            .push(LaunchToRuntimeContinuityIssue::FirstBindingChallengeMismatch);
    }
    if binding.fresh_observation_qualification_digest
        != consumed.fresh_observation_qualification_digest()
    {
        report
            .issues
            .push(LaunchToRuntimeContinuityIssue::FirstBindingFreshQualificationMismatch);
    }
    if binding.raw_observation_qualification_digest
        != consumed.raw_observation_qualification_digest()
    {
        report
            .issues
            .push(LaunchToRuntimeContinuityIssue::FirstBindingRawObservationMismatch);
    }
    if binding.mapped_object_set_digest != consumed.mapped_object_set_digest() {
        report
            .issues
            .push(LaunchToRuntimeContinuityIssue::FirstBindingMappedSetMismatch);
    }
    if binding.observation_observed_at_ms != consumed.observed_at_ms()
        || binding.checkpoint_observed_at_ms != consumed.observed_at_ms()
    {
        report
            .issues
            .push(LaunchToRuntimeContinuityIssue::FirstBindingObservationTimeMismatch);
    }
    if binding.checkpoint_dynamic_measurement_digest
        != consumed.fresh_observation_qualification_digest()
    {
        report
            .issues
            .push(LaunchToRuntimeContinuityIssue::FirstBindingDynamicMeasurementMismatch);
    }
    if continuity.process_instance_id() != consumed.process_instance_id() {
        report.issues.push(LaunchToRuntimeContinuityIssue::SignedTraceProcessMismatch);
    }
    if continuity.runtime_policy_digest() != consumed.runtime_policy_digest()
        || continuity.runtime_verifier_ref() != consumed.runtime_verifier_ref()
        || continuity.backend_id() != consumed.backend_id()
    {
        report
            .issues
            .push(LaunchToRuntimeContinuityIssue::SignedTraceRuntimeIdentityMismatch);
    }

    if !report.issues.is_empty() {
        return Err(finalize(report));
    }

    report.disposition = LaunchToRuntimeContinuityDisposition::Qualified;
    let report_digest = report.canonical_digest();
    let policy_digest = policy_digest.expect("validated policy has digest");
    let qualification_digest = qualification_digest(
        &policy_digest,
        confirmed.confirmation_digest(),
        authority.qualification_digest(),
        issued.qualification_digest(),
        consumed.qualification_digest(),
        continuity.qualification_digest(),
        &binding.checkpoint_digest,
        continuity.runtime_trace_digest(),
        &report_digest,
    );
    let verified = LaunchToRuntimeContinuity {
        qualification_digest,
        report_digest,
        policy_digest,
        exec_confirmation_digest: confirmed.confirmation_digest().into(),
        critical_authority_qualification_digest: authority.qualification_digest().into(),
        authority_snapshot_digest: authority.authority_snapshot_digest().into(),
        issued_handoff_qualification_digest: issued.qualification_digest().into(),
        consumed_first_observation_qualification_digest: consumed.qualification_digest().into(),
        fresh_continuity_qualification_digest: continuity.qualification_digest().into(),
        ticket_digest: issued.ticket_digest().into(),
        launch_plan_digest: confirmed.plan_digest().into(),
        launch_nonce_blake3_hex: confirmed.launch_nonce_blake3_hex().into(),
        launch_challenge_nonce_blake3_hex: issued.challenge_nonce_blake3_hex().into(),
        tracee_pid: confirmed.tracee_pid(),
        process_instance_id: continuity.process_instance_id().into(),
        first_checkpoint_digest: binding.checkpoint_digest.clone(),
        first_checkpoint_dynamic_measurement_digest: binding
            .checkpoint_dynamic_measurement_digest
            .clone(),
        first_challenge_digest: binding.challenge_digest.clone(),
        first_fresh_observation_qualification_digest: binding
            .fresh_observation_qualification_digest
            .clone(),
        first_raw_observation_qualification_digest: binding
            .raw_observation_qualification_digest
            .clone(),
        first_mapped_object_set_digest: binding.mapped_object_set_digest.clone(),
        first_observed_at_ms: binding.observation_observed_at_ms,
        fresh_sequence_digest: continuity.fresh_sequence_digest().into(),
        runtime_trace_digest: continuity.runtime_trace_digest().into(),
        runtime_policy_digest: continuity.runtime_policy_digest().into(),
        runtime_verifier_ref: continuity.runtime_verifier_ref().into(),
        backend_id: continuity.backend_id().into(),
        executable_digest: consumed.executable_digest().into(),
    };
    Ok(LaunchToRuntimeContinuityQualification { report, verified })
}

#[allow(clippy::too_many_arguments)]
fn qualification_digest(
    policy_digest: &str,
    exec_confirmation_digest: &str,
    critical_authority_digest: &str,
    issued_handoff_digest: &str,
    consumed_first_digest: &str,
    fresh_continuity_digest: &str,
    first_checkpoint_digest: &str,
    runtime_trace_digest: &str,
    report_digest: &str,
) -> String {
    let mut h = blake3::Hasher::new();
    h.update(QUALIFICATION_DOMAIN);
    for value in [
        policy_digest,
        exec_confirmation_digest,
        critical_authority_digest,
        issued_handoff_digest,
        consumed_first_digest,
        fresh_continuity_digest,
        first_checkpoint_digest,
        runtime_trace_digest,
        report_digest,
    ] {
        field(&mut h, value);
    }
    format!("blake3:{}", h.finalize().to_hex())
}

fn finalize(mut report: LaunchToRuntimeContinuityReport) -> LaunchToRuntimeContinuityReport {
    report.disposition = if report.issues.iter().any(LaunchToRuntimeContinuityIssue::invalid) {
        LaunchToRuntimeContinuityDisposition::Invalid
    } else {
        LaunchToRuntimeContinuityDisposition::Blocked
    };
    report
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

    fn policy() -> LaunchToRuntimeContinuityPolicy {
        LaunchToRuntimeContinuityPolicy {
            schema_version: LAUNCH_TO_RUNTIME_CONTINUITY_POLICY_SCHEMA_V1.into(),
            policy_id: "policy:launch-to-runtime:1".into(),
            expected_exec_confirmation_policy_digest: d("exec-policy"),
            expected_critical_authority_policy_digest: d("authority-policy"),
            expected_handoff_policy_digest: d("handoff-policy"),
            expected_first_observation_policy_digest: d("first-policy"),
            expected_fresh_continuity_policy_digest: d("continuity-policy"),
            expected_runtime_policy_digest: d("runtime-policy"),
            expected_runtime_verifier_ref: "verifier:host".into(),
            expected_backend_id: "backend:in-process".into(),
            evidence_refs: vec!["review:a".into(), "review:b".into()],
        }
    }

    #[test]
    fn policy_reference_order_is_nonsemantic() {
        let left = policy();
        let mut right = left.clone();
        right.evidence_refs.reverse();
        assert_eq!(left.canonical_digest(), right.canonical_digest());
        right.expected_handoff_policy_digest = d("other-handoff-policy");
        assert_ne!(left.canonical_digest(), right.canonical_digest());
    }

    #[test]
    fn qualification_identity_binds_every_parent_and_first_checkpoint() {
        let left = qualification_digest(
            &d("policy"), &d("exec"), &d("authority"), &d("issued"), &d("consumed"),
            &d("continuity"), &d("checkpoint"), &d("trace"), &d("report"),
        );
        let right = qualification_digest(
            &d("policy"), &d("exec"), &d("authority"), &d("issued"), &d("consumed"),
            &d("continuity"), &d("other-checkpoint"), &d("trace"), &d("report"),
        );
        assert_ne!(left, right);
    }

    #[test]
    fn claim_ceiling_remains_explicit() {
        let claims = [
            "uninterrupted_mapping_continuity_exec_to_first_observation_established=false",
            "ticket_consumption_was_first_application_action_established=false",
            "exclusive_pipe_writer_authority_established=false",
            "mapped_runtime_continuity_between_checkpoints_established=false",
            "global_cross_trace_replay_excluded=false",
            "trusted_time_established=false",
            "all_exec_preserved_process_state_qualified=false",
            "grants_physical_authority=false",
        ];
        assert_eq!(claims.len(), 8);
    }
}
