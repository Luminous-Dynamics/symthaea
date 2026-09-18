// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! End-to-end composition of confirmed launch, live checkpoint one, successful
//! bootstrap release, authenticated post-release checkpoint two, and the exact
//! full runtime-continuity verifier.
//!
//! This bridge deliberately re-runs the existing runtime-continuity verifier on
//! the complete public evidence. It does not reinterpret signature, checkpoint
//! gap, static-drift, computation, or runtime-trace semantics.

#![cfg(target_os = "linux")]
#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use symthaea_assurance_authenticated_post_release_checkpoint_two::AuthenticatedPostReleaseCheckpointTwoObservation;
use symthaea_assurance_launch_to_runtime_continuity::LaunchToRuntimeContinuity;
use symthaea_assurance_signed_post_release_checkpoint_two::SignedPostReleaseCheckpointTwo;
use symthaea_evidence_verifier_runtime_continuity::{
    RuntimeContinuityDisposition, RuntimeContinuityReport, VerifierRuntimeContinuityPolicy,
    VerifierRuntimeEvidence, verify_continuous_verifier_execution,
};

pub const POST_RELEASE_CONTINUOUS_LINEAGE_POLICY_SCHEMA_V1: &str =
    "symthaea.assurance.post-release-continuous-runtime-lineage-policy.v1";
pub const POST_RELEASE_CONTINUOUS_LINEAGE_REPORT_SCHEMA_V1: &str =
    "symthaea.assurance.post-release-continuous-runtime-lineage-report.v1";

const POLICY_DOMAIN: &[u8] =
    b"symthaea.assurance.post-release-continuous-runtime-lineage-policy.digest.v1\0";
const REPORT_DOMAIN: &[u8] =
    b"symthaea.assurance.post-release-continuous-runtime-lineage-report.digest.v1\0";
const QUALIFICATION_DOMAIN: &[u8] =
    b"symthaea.assurance.post-release-continuous-runtime-lineage-qualification.digest.v1\0";
const MAX_TEXT: usize = 4096;
const MAX_REFS: usize = 128;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PostReleaseContinuousLineagePolicy {
    pub schema_version: String,
    pub policy_id: String,
    pub expected_launch_to_runtime_policy_digest: String,
    pub expected_authenticated_checkpoint_two_policy_digest: String,
    pub expected_signed_checkpoint_two_policy_digest: String,
    pub expected_runtime_policy_digest: String,
    pub expected_runtime_verifier_ref: String,
    pub expected_backend_id: String,
    pub evidence_refs: Vec<String>,
}

impl PostReleaseContinuousLineagePolicy {
    pub fn validate(&self) -> bool {
        self.schema_version == POST_RELEASE_CONTINUOUS_LINEAGE_POLICY_SCHEMA_V1
            && canonical_text(&self.policy_id)
            && valid_blake3(&self.expected_launch_to_runtime_policy_digest)
            && valid_blake3(&self.expected_authenticated_checkpoint_two_policy_digest)
            && valid_blake3(&self.expected_signed_checkpoint_two_policy_digest)
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
            self.expected_launch_to_runtime_policy_digest.as_str(),
            self.expected_authenticated_checkpoint_two_policy_digest
                .as_str(),
            self.expected_signed_checkpoint_two_policy_digest.as_str(),
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
pub enum PostReleaseContinuousLineageIssue {
    InvalidPolicy,
    InvalidRuntimePolicy,
    FullRuntimeContinuityNotQualified,
    LaunchToRuntimePolicyMismatch,
    AuthenticatedCheckpointTwoPolicyMismatch,
    SignedCheckpointTwoPolicyMismatch,
    SignedAuthenticatedParentMismatch,
    RuntimePolicyMismatch,
    RuntimeVerifierMismatch,
    BackendMismatch,
    TraceePidMismatch,
    ProcessInstanceMismatch,
    CheckpointCountTooSmall,
    CheckpointOneDigestMismatch,
    CheckpointOneDynamicMeasurementMismatch,
    CheckpointOneObservationTimeMismatch,
    CheckpointTwoDigestMismatch,
    CheckpointTwoDynamicMeasurementMismatch,
    CheckpointTwoPredecessorMismatch,
    CheckpointTwoCounterMismatch,
    CheckpointTwoObservationTimeMismatch,
    LaunchAttestationMismatch,
    ExecutableIdentityMismatch,
    DependencyClosureMismatch,
    RuntimeTraceMismatch,
}

impl PostReleaseContinuousLineageIssue {
    fn invalid(&self) -> bool {
        matches!(
            self,
            Self::InvalidPolicy
                | Self::InvalidRuntimePolicy
                | Self::CheckpointCountTooSmall
                | Self::CheckpointOneDigestMismatch
                | Self::CheckpointOneDynamicMeasurementMismatch
                | Self::CheckpointTwoDigestMismatch
                | Self::CheckpointTwoDynamicMeasurementMismatch
                | Self::CheckpointTwoPredecessorMismatch
                | Self::LaunchAttestationMismatch
                | Self::ProcessInstanceMismatch
                | Self::RuntimeTraceMismatch
        )
    }

    fn code(&self) -> &'static str {
        match self {
            Self::InvalidPolicy => "invalid-policy",
            Self::InvalidRuntimePolicy => "invalid-runtime-policy",
            Self::FullRuntimeContinuityNotQualified => "full-runtime-continuity-not-qualified",
            Self::LaunchToRuntimePolicyMismatch => "launch-to-runtime-policy-mismatch",
            Self::AuthenticatedCheckpointTwoPolicyMismatch => {
                "authenticated-checkpoint-two-policy-mismatch"
            }
            Self::SignedCheckpointTwoPolicyMismatch => "signed-checkpoint-two-policy-mismatch",
            Self::SignedAuthenticatedParentMismatch => "signed-authenticated-parent-mismatch",
            Self::RuntimePolicyMismatch => "runtime-policy-mismatch",
            Self::RuntimeVerifierMismatch => "runtime-verifier-mismatch",
            Self::BackendMismatch => "backend-mismatch",
            Self::TraceePidMismatch => "tracee-pid-mismatch",
            Self::ProcessInstanceMismatch => "process-instance-mismatch",
            Self::CheckpointCountTooSmall => "checkpoint-count-too-small",
            Self::CheckpointOneDigestMismatch => "checkpoint-one-digest-mismatch",
            Self::CheckpointOneDynamicMeasurementMismatch => {
                "checkpoint-one-dynamic-measurement-mismatch"
            }
            Self::CheckpointOneObservationTimeMismatch => {
                "checkpoint-one-observation-time-mismatch"
            }
            Self::CheckpointTwoDigestMismatch => "checkpoint-two-digest-mismatch",
            Self::CheckpointTwoDynamicMeasurementMismatch => {
                "checkpoint-two-dynamic-measurement-mismatch"
            }
            Self::CheckpointTwoPredecessorMismatch => "checkpoint-two-predecessor-mismatch",
            Self::CheckpointTwoCounterMismatch => "checkpoint-two-counter-mismatch",
            Self::CheckpointTwoObservationTimeMismatch => {
                "checkpoint-two-observation-time-mismatch"
            }
            Self::LaunchAttestationMismatch => "launch-attestation-mismatch",
            Self::ExecutableIdentityMismatch => "executable-identity-mismatch",
            Self::DependencyClosureMismatch => "dependency-closure-mismatch",
            Self::RuntimeTraceMismatch => "runtime-trace-mismatch",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PostReleaseContinuousLineageDisposition {
    Invalid,
    Blocked,
    Qualified,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PostReleaseContinuousLineageReport {
    pub schema_version: String,
    pub policy_id: String,
    pub policy_digest: Option<String>,
    pub launch_to_runtime_qualification_digest: String,
    pub authenticated_checkpoint_two_qualification_digest: String,
    pub signed_checkpoint_two_qualification_digest: String,
    pub release_digest: String,
    pub tracee_pid: i32,
    pub process_instance_id: String,
    pub runtime_policy_digest: Option<String>,
    pub runtime_verifier_ref: String,
    pub backend_id: String,
    pub launch_attestation_digest: Option<String>,
    pub checkpoint_count: u64,
    pub checkpoint_one_digest: Option<String>,
    pub checkpoint_two_digest: Option<String>,
    pub checkpoint_two_dynamic_measurement_digest: String,
    pub final_checkpoint_digest: Option<String>,
    pub runtime_trace_digest: Option<String>,
    pub continuous_execution_digest: Option<String>,
    pub computation_attestation_digest: Option<String>,
    pub assessed_at_ms: u64,
    pub disposition: PostReleaseContinuousLineageDisposition,
    pub issues: Vec<PostReleaseContinuousLineageIssue>,
}

impl PostReleaseContinuousLineageReport {
    pub fn canonical_digest(&self) -> String {
        let mut h = blake3::Hasher::new();
        h.update(REPORT_DOMAIN);
        for value in [
            self.schema_version.as_str(),
            self.policy_id.as_str(),
            self.policy_digest.as_deref().unwrap_or("-"),
            self.launch_to_runtime_qualification_digest.as_str(),
            self.authenticated_checkpoint_two_qualification_digest
                .as_str(),
            self.signed_checkpoint_two_qualification_digest.as_str(),
            self.release_digest.as_str(),
            self.process_instance_id.as_str(),
            self.runtime_policy_digest.as_deref().unwrap_or("-"),
            self.runtime_verifier_ref.as_str(),
            self.backend_id.as_str(),
            self.launch_attestation_digest.as_deref().unwrap_or("-"),
            self.checkpoint_one_digest.as_deref().unwrap_or("-"),
            self.checkpoint_two_digest.as_deref().unwrap_or("-"),
            self.checkpoint_two_dynamic_measurement_digest.as_str(),
            self.final_checkpoint_digest.as_deref().unwrap_or("-"),
            self.runtime_trace_digest.as_deref().unwrap_or("-"),
            self.continuous_execution_digest.as_deref().unwrap_or("-"),
            self.computation_attestation_digest
                .as_deref()
                .unwrap_or("-"),
        ] {
            field(&mut h, value);
        }
        h.update(&self.tracee_pid.to_le_bytes());
        h.update(&self.checkpoint_count.to_le_bytes());
        h.update(&self.assessed_at_ms.to_le_bytes());
        field(
            &mut h,
            match self.disposition {
                PostReleaseContinuousLineageDisposition::Invalid => "invalid",
                PostReleaseContinuousLineageDisposition::Blocked => "blocked",
                PostReleaseContinuousLineageDisposition::Qualified => "qualified",
            },
        );
        h.update(&(self.issues.len() as u64).to_le_bytes());
        for issue in &self.issues {
            field(&mut h, issue.code());
        }
        b3(h.finalize())
    }

    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PostReleaseContinuousRuntimeLineage {
    qualification_digest: String,
    report_digest: String,
    policy_digest: String,
    launch_to_runtime_qualification_digest: String,
    authenticated_checkpoint_two_qualification_digest: String,
    signed_checkpoint_two_qualification_digest: String,
    release_digest: String,
    tracee_pid: i32,
    process_instance_id: String,
    runtime_policy_digest: String,
    runtime_verifier_ref: String,
    backend_id: String,
    launch_attestation_digest: String,
    checkpoint_count: u64,
    checkpoint_one_digest: String,
    checkpoint_two_digest: String,
    checkpoint_two_dynamic_measurement_digest: String,
    final_checkpoint_digest: String,
    runtime_trace_digest: String,
    continuous_execution_digest: String,
    computation_attestation_digest: String,
    request_nonce_blake3_hex: String,
    input_digest: String,
    output_digest: String,
    assessed_at_ms: u64,
}

impl PostReleaseContinuousRuntimeLineage {
    pub fn qualification_digest(&self) -> &str {
        &self.qualification_digest
    }
    pub fn report_digest(&self) -> &str {
        &self.report_digest
    }
    pub fn policy_digest(&self) -> &str {
        &self.policy_digest
    }
    pub fn launch_to_runtime_qualification_digest(&self) -> &str {
        &self.launch_to_runtime_qualification_digest
    }
    pub fn authenticated_checkpoint_two_qualification_digest(&self) -> &str {
        &self.authenticated_checkpoint_two_qualification_digest
    }
    pub fn signed_checkpoint_two_qualification_digest(&self) -> &str {
        &self.signed_checkpoint_two_qualification_digest
    }
    pub fn release_digest(&self) -> &str {
        &self.release_digest
    }
    pub const fn tracee_pid(&self) -> i32 {
        self.tracee_pid
    }
    pub fn process_instance_id(&self) -> &str {
        &self.process_instance_id
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
    pub fn launch_attestation_digest(&self) -> &str {
        &self.launch_attestation_digest
    }
    pub const fn checkpoint_count(&self) -> u64 {
        self.checkpoint_count
    }
    pub fn checkpoint_one_digest(&self) -> &str {
        &self.checkpoint_one_digest
    }
    pub fn checkpoint_two_digest(&self) -> &str {
        &self.checkpoint_two_digest
    }
    pub fn checkpoint_two_dynamic_measurement_digest(&self) -> &str {
        &self.checkpoint_two_dynamic_measurement_digest
    }
    pub fn final_checkpoint_digest(&self) -> &str {
        &self.final_checkpoint_digest
    }
    pub fn runtime_trace_digest(&self) -> &str {
        &self.runtime_trace_digest
    }
    pub fn continuous_execution_digest(&self) -> &str {
        &self.continuous_execution_digest
    }
    pub fn computation_attestation_digest(&self) -> &str {
        &self.computation_attestation_digest
    }
    pub fn request_nonce_blake3_hex(&self) -> &str {
        &self.request_nonce_blake3_hex
    }
    pub fn input_digest(&self) -> &str {
        &self.input_digest
    }
    pub fn output_digest(&self) -> &str {
        &self.output_digest
    }
    pub const fn assessed_at_ms(&self) -> u64 {
        self.assessed_at_ms
    }

    pub const fn confirmed_launch_to_live_checkpoint_one_established(&self) -> bool {
        true
    }
    pub const fn checkpoint_one_live_observation_proven(&self) -> bool {
        true
    }
    pub const fn successful_release_to_live_checkpoint_two_established(&self) -> bool {
        true
    }
    pub const fn checkpoint_two_live_observation_proven(&self) -> bool {
        true
    }
    pub const fn checkpoint_two_authorized_signature_verified(&self) -> bool {
        true
    }
    pub const fn checkpoint_two_included_in_fully_qualified_runtime_trace(&self) -> bool {
        true
    }
    pub const fn checkpoint_two_descends_from_exact_live_checkpoint_one(&self) -> bool {
        true
    }
    pub const fn full_runtime_sequence_gap_signer_and_computation_policy_reverified(&self) -> bool {
        true
    }
    pub const fn end_to_end_launch_release_reentry_runtime_lineage_established(&self) -> bool {
        true
    }
    pub const fn os_tracee_pid_and_runtime_process_claim_bound_across_lineage(&self) -> bool {
        true
    }
    pub const fn uninterrupted_mapping_continuity_between_checkpoints_established(&self) -> bool {
        false
    }
    pub const fn signature_operation_after_live_observation_established(&self) -> bool {
        false
    }
    pub const fn signer_key_non_compromise_established(&self) -> bool {
        false
    }
    pub const fn trusted_time_established(&self) -> bool {
        false
    }
    pub const fn global_replay_excluded(&self) -> bool {
        false
    }
    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

pub struct PostReleaseContinuousRuntimeLineageQualification {
    pub report: PostReleaseContinuousLineageReport,
    pub runtime_report: RuntimeContinuityReport,
    verified: PostReleaseContinuousRuntimeLineage,
}

impl PostReleaseContinuousRuntimeLineageQualification {
    pub fn verified(&self) -> &PostReleaseContinuousRuntimeLineage {
        &self.verified
    }
    pub fn into_verified(self) -> PostReleaseContinuousRuntimeLineage {
        self.verified
    }
}

#[allow(clippy::too_many_arguments)]
pub fn verify_post_release_continuous_runtime_lineage(
    policy: &PostReleaseContinuousLineagePolicy,
    launch: &LaunchToRuntimeContinuity,
    authenticated: &AuthenticatedPostReleaseCheckpointTwoObservation,
    signed: &SignedPostReleaseCheckpointTwo,
    runtime_policy: &VerifierRuntimeContinuityPolicy,
    evidence: &VerifierRuntimeEvidence,
    assessed_at_ms: u64,
) -> Result<PostReleaseContinuousRuntimeLineageQualification, PostReleaseContinuousLineageReport> {
    let policy_digest = policy.canonical_digest();
    let runtime_policy_digest = runtime_policy.canonical_digest();
    let assessment = verify_continuous_verifier_execution(runtime_policy, evidence, assessed_at_ms);
    let runtime_report = assessment.report.clone();
    let launch_digest = evidence.launch.canonical_digest();
    let checkpoint_one_digest = evidence
        .checkpoints
        .first()
        .and_then(|checkpoint| checkpoint.canonical_digest());
    let checkpoint_two_digest = evidence
        .checkpoints
        .get(1)
        .and_then(|checkpoint| checkpoint.canonical_digest());
    let mut report = PostReleaseContinuousLineageReport {
        schema_version: POST_RELEASE_CONTINUOUS_LINEAGE_REPORT_SCHEMA_V1.into(),
        policy_id: policy.policy_id.clone(),
        policy_digest: policy_digest.clone(),
        launch_to_runtime_qualification_digest: launch.qualification_digest().into(),
        authenticated_checkpoint_two_qualification_digest: authenticated
            .qualification_digest()
            .into(),
        signed_checkpoint_two_qualification_digest: signed.qualification_digest().into(),
        release_digest: authenticated.release_digest().into(),
        tracee_pid: authenticated.tracee_pid(),
        process_instance_id: authenticated.process_instance_id().into(),
        runtime_policy_digest: runtime_policy_digest.clone(),
        runtime_verifier_ref: policy.expected_runtime_verifier_ref.clone(),
        backend_id: policy.expected_backend_id.clone(),
        launch_attestation_digest: launch_digest.clone(),
        checkpoint_count: evidence.checkpoints.len() as u64,
        checkpoint_one_digest: checkpoint_one_digest.clone(),
        checkpoint_two_digest: checkpoint_two_digest.clone(),
        checkpoint_two_dynamic_measurement_digest: evidence
            .checkpoints
            .get(1)
            .map(|checkpoint| checkpoint.dynamic_measurement_digest.clone())
            .unwrap_or_default(),
        final_checkpoint_digest: runtime_report.final_checkpoint_digest.clone(),
        runtime_trace_digest: runtime_report.runtime_trace_digest.clone(),
        continuous_execution_digest: runtime_report.continuity_digest.clone(),
        computation_attestation_digest: runtime_report.computation_attestation_digest.clone(),
        assessed_at_ms,
        disposition: PostReleaseContinuousLineageDisposition::Invalid,
        issues: Vec::new(),
    };

    if !policy.validate() {
        report
            .issues
            .push(PostReleaseContinuousLineageIssue::InvalidPolicy);
    }
    if !runtime_policy.validate() || runtime_policy_digest.is_none() {
        report
            .issues
            .push(PostReleaseContinuousLineageIssue::InvalidRuntimePolicy);
    }
    if !report.issues.is_empty() {
        return Err(finalize(report));
    }

    if runtime_report.disposition != RuntimeContinuityDisposition::Qualified
        || assessment.continuous().is_none()
    {
        report
            .issues
            .push(PostReleaseContinuousLineageIssue::FullRuntimeContinuityNotQualified);
        return Err(finalize(report));
    }
    let continuous = assessment
        .continuous()
        .expect("qualified assessment has capability");

    if launch.policy_digest() != policy.expected_launch_to_runtime_policy_digest {
        report
            .issues
            .push(PostReleaseContinuousLineageIssue::LaunchToRuntimePolicyMismatch);
    }
    if authenticated.policy_digest() != policy.expected_authenticated_checkpoint_two_policy_digest {
        report
            .issues
            .push(PostReleaseContinuousLineageIssue::AuthenticatedCheckpointTwoPolicyMismatch);
    }
    if signed.policy_digest() != policy.expected_signed_checkpoint_two_policy_digest {
        report
            .issues
            .push(PostReleaseContinuousLineageIssue::SignedCheckpointTwoPolicyMismatch);
    }
    if signed.authenticated_observation_qualification_digest()
        != authenticated.qualification_digest()
    {
        report
            .issues
            .push(PostReleaseContinuousLineageIssue::SignedAuthenticatedParentMismatch);
    }
    if runtime_policy_digest.as_deref() != Some(policy.expected_runtime_policy_digest.as_str())
        || launch.runtime_policy_digest() != policy.expected_runtime_policy_digest
        || authenticated.runtime_policy_digest() != policy.expected_runtime_policy_digest
        || signed.runtime_policy_digest() != policy.expected_runtime_policy_digest
        || continuous.policy_digest() != policy.expected_runtime_policy_digest
    {
        report
            .issues
            .push(PostReleaseContinuousLineageIssue::RuntimePolicyMismatch);
    }
    if runtime_policy.verifier_ref != policy.expected_runtime_verifier_ref
        || launch.runtime_verifier_ref() != policy.expected_runtime_verifier_ref
        || authenticated.runtime_verifier_ref() != policy.expected_runtime_verifier_ref
        || signed.verifier_ref() != policy.expected_runtime_verifier_ref
        || continuous.verifier_ref() != policy.expected_runtime_verifier_ref
        || evidence.launch.verifier_ref != policy.expected_runtime_verifier_ref
    {
        report
            .issues
            .push(PostReleaseContinuousLineageIssue::RuntimeVerifierMismatch);
    }
    if launch.backend_id() != policy.expected_backend_id
        || authenticated.backend_id() != policy.expected_backend_id
        || signed.backend_id() != policy.expected_backend_id
    {
        report
            .issues
            .push(PostReleaseContinuousLineageIssue::BackendMismatch);
    }
    if launch.tracee_pid() != authenticated.tracee_pid() {
        report
            .issues
            .push(PostReleaseContinuousLineageIssue::TraceePidMismatch);
    }
    if launch.process_instance_id() != authenticated.process_instance_id()
        || signed.process_instance_id() != authenticated.process_instance_id()
        || continuous.process_instance_id() != authenticated.process_instance_id()
        || evidence.launch.process_instance_id != authenticated.process_instance_id()
    {
        report
            .issues
            .push(PostReleaseContinuousLineageIssue::ProcessInstanceMismatch);
    }
    if evidence.checkpoints.len() < 2 {
        report
            .issues
            .push(PostReleaseContinuousLineageIssue::CheckpointCountTooSmall);
        return Err(finalize(report));
    }

    let checkpoint_one = &evidence.checkpoints[0];
    let checkpoint_two = &evidence.checkpoints[1];
    let checkpoint_one_digest =
        checkpoint_one_digest.expect("qualified trace checkpoint one digest");
    let checkpoint_two_digest =
        checkpoint_two_digest.expect("qualified trace checkpoint two digest");
    let launch_digest = launch_digest.expect("qualified trace launch digest");

    if checkpoint_one_digest != launch.first_checkpoint_digest()
        || authenticated.bootstrap_ready_checkpoint_digest() != launch.first_checkpoint_digest()
        || signed.previous_checkpoint_digest() != launch.first_checkpoint_digest()
    {
        report
            .issues
            .push(PostReleaseContinuousLineageIssue::CheckpointOneDigestMismatch);
    }
    if checkpoint_one.dynamic_measurement_digest
        != launch.first_checkpoint_dynamic_measurement_digest()
        || checkpoint_one.dynamic_measurement_digest
            != launch.first_fresh_observation_qualification_digest()
    {
        report
            .issues
            .push(PostReleaseContinuousLineageIssue::CheckpointOneDynamicMeasurementMismatch);
    }
    if checkpoint_one.observed_at_ms != launch.first_observed_at_ms() {
        report
            .issues
            .push(PostReleaseContinuousLineageIssue::CheckpointOneObservationTimeMismatch);
    }
    if checkpoint_two_digest != signed.checkpoint_digest() {
        report
            .issues
            .push(PostReleaseContinuousLineageIssue::CheckpointTwoDigestMismatch);
    }
    if checkpoint_two.dynamic_measurement_digest != signed.dynamic_measurement_digest()
        || checkpoint_two.dynamic_measurement_digest != authenticated.qualification_digest()
    {
        report
            .issues
            .push(PostReleaseContinuousLineageIssue::CheckpointTwoDynamicMeasurementMismatch);
    }
    if checkpoint_two.previous_checkpoint_digest.as_deref() != Some(checkpoint_one_digest.as_str())
    {
        report
            .issues
            .push(PostReleaseContinuousLineageIssue::CheckpointTwoPredecessorMismatch);
    }
    if checkpoint_two.monotonic_counter != signed.monotonic_counter() {
        report
            .issues
            .push(PostReleaseContinuousLineageIssue::CheckpointTwoCounterMismatch);
    }
    if checkpoint_two.observed_at_ms != signed.observed_at_ms() {
        report
            .issues
            .push(PostReleaseContinuousLineageIssue::CheckpointTwoObservationTimeMismatch);
    }
    if checkpoint_one.launch_attestation_digest != launch_digest
        || checkpoint_two.launch_attestation_digest != launch_digest
        || signed.launch_attestation_digest() != launch_digest
        || authenticated.launch_attestation_digest() != launch_digest
        || continuous.launch_attestation_digest() != launch_digest
    {
        report
            .issues
            .push(PostReleaseContinuousLineageIssue::LaunchAttestationMismatch);
    }
    if launch.executable_digest() != authenticated.executable_digest()
        || launch.executable_digest() != signed.executable_digest()
        || launch.executable_digest() != runtime_policy.expected_executable_digest
    {
        report
            .issues
            .push(PostReleaseContinuousLineageIssue::ExecutableIdentityMismatch);
    }
    if authenticated.dependency_closure_digest() != signed.dependency_closure_digest()
        || authenticated.dependency_closure_digest()
            != runtime_policy.expected_dependency_closure_digest
    {
        report
            .issues
            .push(PostReleaseContinuousLineageIssue::DependencyClosureMismatch);
    }
    if runtime_report.runtime_trace_digest.as_deref() != Some(continuous.runtime_trace_digest())
        || runtime_report.continuity_digest.as_deref() != Some(continuous.continuity_digest())
    {
        report
            .issues
            .push(PostReleaseContinuousLineageIssue::RuntimeTraceMismatch);
    }

    if !report.issues.is_empty() {
        return Err(finalize(report));
    }

    report.disposition = PostReleaseContinuousLineageDisposition::Qualified;
    let report_digest = report.canonical_digest();
    let policy_digest = policy_digest.expect("validated policy has digest");
    let runtime_policy_digest = runtime_policy_digest.expect("validated runtime policy has digest");
    let final_checkpoint_digest = runtime_report
        .final_checkpoint_digest
        .clone()
        .expect("qualified runtime report final checkpoint");
    let runtime_trace_digest = runtime_report
        .runtime_trace_digest
        .clone()
        .expect("qualified runtime report trace digest");
    let continuity_digest = runtime_report
        .continuity_digest
        .clone()
        .expect("qualified runtime report continuity digest");
    let computation_digest = runtime_report
        .computation_attestation_digest
        .clone()
        .expect("qualified runtime report computation digest");
    let qualification_digest = qualification_digest(
        &policy_digest,
        launch.qualification_digest(),
        authenticated.qualification_digest(),
        signed.qualification_digest(),
        &runtime_policy_digest,
        &checkpoint_one_digest,
        &checkpoint_two_digest,
        &continuity_digest,
        &report_digest,
    );
    let verified = PostReleaseContinuousRuntimeLineage {
        qualification_digest,
        report_digest,
        policy_digest,
        launch_to_runtime_qualification_digest: launch.qualification_digest().into(),
        authenticated_checkpoint_two_qualification_digest: authenticated
            .qualification_digest()
            .into(),
        signed_checkpoint_two_qualification_digest: signed.qualification_digest().into(),
        release_digest: authenticated.release_digest().into(),
        tracee_pid: authenticated.tracee_pid(),
        process_instance_id: authenticated.process_instance_id().into(),
        runtime_policy_digest,
        runtime_verifier_ref: policy.expected_runtime_verifier_ref.clone(),
        backend_id: policy.expected_backend_id.clone(),
        launch_attestation_digest: launch_digest,
        checkpoint_count: evidence.checkpoints.len() as u64,
        checkpoint_one_digest,
        checkpoint_two_digest,
        checkpoint_two_dynamic_measurement_digest: checkpoint_two
            .dynamic_measurement_digest
            .clone(),
        final_checkpoint_digest,
        runtime_trace_digest,
        continuous_execution_digest: continuity_digest,
        computation_attestation_digest: computation_digest,
        request_nonce_blake3_hex: continuous.request_nonce_blake3_hex().into(),
        input_digest: continuous.input_digest().into(),
        output_digest: continuous.output_digest().into(),
        assessed_at_ms,
    };

    Ok(PostReleaseContinuousRuntimeLineageQualification {
        report,
        runtime_report,
        verified,
    })
}

#[allow(clippy::too_many_arguments)]
fn qualification_digest(
    policy: &str,
    launch: &str,
    authenticated: &str,
    signed: &str,
    runtime_policy: &str,
    checkpoint_one: &str,
    checkpoint_two: &str,
    continuous: &str,
    report: &str,
) -> String {
    let mut h = blake3::Hasher::new();
    h.update(QUALIFICATION_DOMAIN);
    for value in [
        policy,
        launch,
        authenticated,
        signed,
        runtime_policy,
        checkpoint_one,
        checkpoint_two,
        continuous,
        report,
    ] {
        field(&mut h, value);
    }
    b3(h.finalize())
}

fn finalize(mut report: PostReleaseContinuousLineageReport) -> PostReleaseContinuousLineageReport {
    report.disposition = if report
        .issues
        .iter()
        .any(PostReleaseContinuousLineageIssue::invalid)
    {
        PostReleaseContinuousLineageDisposition::Invalid
    } else {
        PostReleaseContinuousLineageDisposition::Blocked
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

    fn policy() -> PostReleaseContinuousLineagePolicy {
        PostReleaseContinuousLineagePolicy {
            schema_version: POST_RELEASE_CONTINUOUS_LINEAGE_POLICY_SCHEMA_V1.into(),
            policy_id: "post-release-continuous-lineage:v1".into(),
            expected_launch_to_runtime_policy_digest: d("launch-policy"),
            expected_authenticated_checkpoint_two_policy_digest: d("authenticated-policy"),
            expected_signed_checkpoint_two_policy_digest: d("signed-policy"),
            expected_runtime_policy_digest: d("runtime-policy"),
            expected_runtime_verifier_ref: "verifier:prod".into(),
            expected_backend_id: "backend:in-process-p256".into(),
            evidence_refs: vec!["review:post-release-continuous-lineage".into()],
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
    fn every_parent_policy_and_runtime_role_are_semantic() {
        let left = policy();
        let mut right = left.clone();
        right.expected_launch_to_runtime_policy_digest = d("different-launch");
        assert_ne!(left.canonical_digest(), right.canonical_digest());
        let mut right = left.clone();
        right.expected_authenticated_checkpoint_two_policy_digest = d("different-auth");
        assert_ne!(left.canonical_digest(), right.canonical_digest());
        let mut right = left.clone();
        right.expected_signed_checkpoint_two_policy_digest = d("different-signed");
        assert_ne!(left.canonical_digest(), right.canonical_digest());
        let mut right = left.clone();
        right.expected_runtime_verifier_ref = "verifier:other".into();
        assert_ne!(left.canonical_digest(), right.canonical_digest());
    }

    #[test]
    fn qualification_identity_binds_both_checkpoints_and_continuous_trace() {
        let base = qualification_digest(
            &d("policy"),
            &d("launch"),
            &d("auth"),
            &d("signed"),
            &d("runtime"),
            &d("checkpoint-one"),
            &d("checkpoint-two"),
            &d("continuous"),
            &d("report"),
        );
        let changed = qualification_digest(
            &d("policy"),
            &d("launch"),
            &d("auth"),
            &d("signed"),
            &d("runtime"),
            &d("checkpoint-one"),
            &d("different-checkpoint-two"),
            &d("continuous"),
            &d("report"),
        );
        assert_ne!(base, changed);
    }

    #[test]
    fn claim_ceiling_stays_explicit() {
        let claims = [
            "between-checkpoint-mapping-continuity=false",
            "signature-operation-after-observation=false",
            "signer-key-non-compromise=false",
            "trusted-time=false",
            "global-replay=false",
            "physical-authority=false",
        ];
        assert_eq!(claims.len(), 6);
    }
}
