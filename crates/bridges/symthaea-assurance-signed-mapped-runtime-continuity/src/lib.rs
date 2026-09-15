// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exact composition of signed runtime checkpoints with mapped-executable observations.
//!
//! The parent runtime-continuity theorem has already verified launch/checkpoint/
//! computation signatures and hash linkage. This bridge independently rebinds
//! the public evidence to that exact opaque trace, then requires every signed
//! checkpoint's `dynamic_measurement_digest` to commit one exact qualified
//! mapped-runtime observation.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use symthaea_assurance_closure_backed_in_process_continuous::
    ClosureBackedInProcessContinuousVerifierExecution;
use symthaea_assurance_observed_mapped_nix_executable_runtime::
    ObservedMappedNixExecutableRuntime;
use symthaea_evidence_verifier_runtime_continuity::{
    ContinuousVerifierExecution, VerifierRuntimeEvidence,
};

pub const SIGNED_MAPPED_RUNTIME_POLICY_SCHEMA_V1: &str =
    "symthaea.assurance.signed-mapped-runtime-continuity-policy.v1";
pub const SIGNED_MAPPED_RUNTIME_REPORT_SCHEMA_V1: &str =
    "symthaea.assurance.signed-mapped-runtime-continuity-report.v1";

const POLICY_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.signed-mapped-runtime-continuity-policy.digest.v1\0";
const RUNTIME_TRACE_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.verifier-runtime-trace.digest.v1\0";
const CHECKPOINT_MAPPING_SEQUENCE_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.signed-mapped-runtime-checkpoint-sequence.digest.v1\0";
const REPORT_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.signed-mapped-runtime-continuity-report.digest.v1\0";
const QUALIFICATION_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.signed-mapped-runtime-continuity-qualification.digest.v1\0";
const MAX_TEXT_BYTES: usize = 4096;
const MAX_EVIDENCE_REFS: usize = 128;
const MAX_CHECKPOINT_BINDINGS: usize = 65_536;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignedMappedRuntimePolicy {
    pub schema_version: String,
    pub policy_id: String,
    pub expected_closure_backed_policy_digest: String,
    pub expected_mapped_runtime_policy_digest: String,
    pub expected_runtime_policy_digest: String,
    pub expected_runtime_verifier_ref: String,
    pub expected_backend_id: String,
    pub evidence_refs: Vec<String>,
}

impl SignedMappedRuntimePolicy {
    pub fn validate(&self) -> bool {
        self.schema_version == SIGNED_MAPPED_RUNTIME_POLICY_SCHEMA_V1
            && canonical_text(&self.policy_id)
            && valid_blake3_digest(&self.expected_closure_backed_policy_digest)
            && valid_blake3_digest(&self.expected_mapped_runtime_policy_digest)
            && valid_blake3_digest(&self.expected_runtime_policy_digest)
            && canonical_text(&self.expected_runtime_verifier_ref)
            && canonical_text(&self.expected_backend_id)
            && valid_refs(&self.evidence_refs)
    }

    pub fn canonical_digest(&self) -> Option<String> {
        if !self.validate() {
            return None;
        }
        let mut hasher = blake3::Hasher::new();
        hasher.update(POLICY_DIGEST_DOMAIN);
        for field in [
            self.schema_version.as_str(),
            self.policy_id.as_str(),
            self.expected_closure_backed_policy_digest.as_str(),
            self.expected_mapped_runtime_policy_digest.as_str(),
            self.expected_runtime_policy_digest.as_str(),
            self.expected_runtime_verifier_ref.as_str(),
            self.expected_backend_id.as_str(),
        ] {
            push_field_le(&mut hasher, field);
        }
        push_sorted_refs_le(&mut hasher, &self.evidence_refs);
        Some(format!("blake3:{}", hasher.finalize().to_hex()))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignedMappedRuntimeDisposition {
    Invalid,
    Blocked,
    Qualified,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignedMappedRuntimeIssue {
    InvalidPolicy,
    InvalidRuntimeEvidence,
    ClosureBackedPolicyMismatch,
    ContinuousExecutionMismatch,
    RuntimePolicyMismatch,
    RuntimeVerifierMismatch,
    BackendMismatch,
    ProcessInstanceMismatch,
    RuntimeTraceMismatch,
    InputDigestMismatch,
    OutputDigestMismatch,
    ComputationTimeMismatch,
    AssessmentTimeMismatch,
    LaunchDigestMismatch,
    FinalCheckpointDigestMismatch,
    ComputationDigestMismatch,
    ObservationCountMismatch { checkpoints: u64, observations: u64 },
    MappedRuntimePolicyMismatch { sequence: u64 },
    NixBindingQualificationMismatch { sequence: u64 },
    ObservationRuntimePolicyMismatch { sequence: u64 },
    ObservationVerifierMismatch { sequence: u64 },
    ObservationBackendMismatch { sequence: u64 },
    ObservationHostIdentityMismatch { sequence: u64 },
    ObservationExecutablePathMismatch { sequence: u64 },
    ObservationExecutableDigestMismatch { sequence: u64 },
    ObservationClosureDigestMismatch { sequence: u64 },
    ObservationTimeMismatch { sequence: u64 },
    CheckpointVerifierMismatch { sequence: u64 },
    CheckpointProcessMismatch { sequence: u64 },
    CheckpointExecutableDigestMismatch { sequence: u64 },
    CheckpointClosureDigestMismatch { sequence: u64 },
    DynamicMeasurementDigestMismatch { sequence: u64 },
}

impl SignedMappedRuntimeIssue {
    fn is_invalid(&self) -> bool {
        matches!(self, Self::InvalidPolicy | Self::InvalidRuntimeEvidence)
    }

    fn code(&self) -> String {
        match self {
            Self::InvalidPolicy => "invalid-policy".into(),
            Self::InvalidRuntimeEvidence => "invalid-runtime-evidence".into(),
            Self::ClosureBackedPolicyMismatch => "closure-backed-policy-mismatch".into(),
            Self::ContinuousExecutionMismatch => "continuous-execution-mismatch".into(),
            Self::RuntimePolicyMismatch => "runtime-policy-mismatch".into(),
            Self::RuntimeVerifierMismatch => "runtime-verifier-mismatch".into(),
            Self::BackendMismatch => "backend-mismatch".into(),
            Self::ProcessInstanceMismatch => "process-instance-mismatch".into(),
            Self::RuntimeTraceMismatch => "runtime-trace-mismatch".into(),
            Self::InputDigestMismatch => "input-digest-mismatch".into(),
            Self::OutputDigestMismatch => "output-digest-mismatch".into(),
            Self::ComputationTimeMismatch => "computation-time-mismatch".into(),
            Self::AssessmentTimeMismatch => "assessment-time-mismatch".into(),
            Self::LaunchDigestMismatch => "launch-digest-mismatch".into(),
            Self::FinalCheckpointDigestMismatch => "final-checkpoint-digest-mismatch".into(),
            Self::ComputationDigestMismatch => "computation-digest-mismatch".into(),
            Self::ObservationCountMismatch { checkpoints, observations } => {
                format!("observation-count-mismatch:{checkpoints}:{observations}")
            }
            Self::MappedRuntimePolicyMismatch { sequence } => {
                format!("mapped-runtime-policy-mismatch:{sequence}")
            }
            Self::NixBindingQualificationMismatch { sequence } => {
                format!("nix-binding-qualification-mismatch:{sequence}")
            }
            Self::ObservationRuntimePolicyMismatch { sequence } => {
                format!("observation-runtime-policy-mismatch:{sequence}")
            }
            Self::ObservationVerifierMismatch { sequence } => {
                format!("observation-verifier-mismatch:{sequence}")
            }
            Self::ObservationBackendMismatch { sequence } => {
                format!("observation-backend-mismatch:{sequence}")
            }
            Self::ObservationHostIdentityMismatch { sequence } => {
                format!("observation-host-identity-mismatch:{sequence}")
            }
            Self::ObservationExecutablePathMismatch { sequence } => {
                format!("observation-executable-path-mismatch:{sequence}")
            }
            Self::ObservationExecutableDigestMismatch { sequence } => {
                format!("observation-executable-digest-mismatch:{sequence}")
            }
            Self::ObservationClosureDigestMismatch { sequence } => {
                format!("observation-closure-digest-mismatch:{sequence}")
            }
            Self::ObservationTimeMismatch { sequence } => {
                format!("observation-time-mismatch:{sequence}")
            }
            Self::CheckpointVerifierMismatch { sequence } => {
                format!("checkpoint-verifier-mismatch:{sequence}")
            }
            Self::CheckpointProcessMismatch { sequence } => {
                format!("checkpoint-process-mismatch:{sequence}")
            }
            Self::CheckpointExecutableDigestMismatch { sequence } => {
                format!("checkpoint-executable-digest-mismatch:{sequence}")
            }
            Self::CheckpointClosureDigestMismatch { sequence } => {
                format!("checkpoint-closure-digest-mismatch:{sequence}")
            }
            Self::DynamicMeasurementDigestMismatch { sequence } => {
                format!("dynamic-measurement-digest-mismatch:{sequence}")
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignedMappedCheckpointBinding {
    pub sequence: u64,
    pub checkpoint_digest: String,
    pub checkpoint_dynamic_measurement_digest: String,
    pub observation_qualification_digest: String,
    pub mapped_object_set_digest: String,
    pub checkpoint_observed_at_ms: u64,
    pub observation_observed_at_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignedMappedRuntimeReport {
    pub schema_version: String,
    pub policy_id: String,
    pub policy_digest: Option<String>,
    pub closure_backed_policy_digest: String,
    pub closure_backed_qualification_digest: String,
    pub continuous_execution_digest: String,
    pub runtime_policy_digest: String,
    pub runtime_verifier_ref: String,
    pub backend_id: String,
    pub process_instance_id: String,
    pub runtime_trace_digest: String,
    pub nix_binding_qualification_digest: String,
    pub host_identity_digest: String,
    pub executable_path: String,
    pub executable_digest: String,
    pub dependency_closure_digest: String,
    pub checkpoint_mapping_sequence_digest: Option<String>,
    pub checkpoint_count: u64,
    pub first_observed_at_ms: u64,
    pub last_observed_at_ms: u64,
    pub disposition: SignedMappedRuntimeDisposition,
    pub issues: Vec<SignedMappedRuntimeIssue>,
}

impl SignedMappedRuntimeReport {
    pub fn canonical_digest(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(REPORT_DIGEST_DOMAIN);
        for field in [
            self.schema_version.as_str(),
            self.policy_id.as_str(),
            self.policy_digest.as_deref().unwrap_or("-"),
            self.closure_backed_policy_digest.as_str(),
            self.closure_backed_qualification_digest.as_str(),
            self.continuous_execution_digest.as_str(),
            self.runtime_policy_digest.as_str(),
            self.runtime_verifier_ref.as_str(),
            self.backend_id.as_str(),
            self.process_instance_id.as_str(),
            self.runtime_trace_digest.as_str(),
            self.nix_binding_qualification_digest.as_str(),
            self.host_identity_digest.as_str(),
            self.executable_path.as_str(),
            self.executable_digest.as_str(),
            self.dependency_closure_digest.as_str(),
            self.checkpoint_mapping_sequence_digest.as_deref().unwrap_or("-"),
        ] {
            push_field_le(&mut hasher, field);
        }
        hasher.update(&self.checkpoint_count.to_le_bytes());
        hasher.update(&self.first_observed_at_ms.to_le_bytes());
        hasher.update(&self.last_observed_at_ms.to_le_bytes());
        push_field_le(
            &mut hasher,
            match self.disposition {
                SignedMappedRuntimeDisposition::Invalid => "invalid",
                SignedMappedRuntimeDisposition::Blocked => "blocked",
                SignedMappedRuntimeDisposition::Qualified => "qualified",
            },
        );
        hasher.update(&(self.issues.len() as u64).to_le_bytes());
        for issue in &self.issues {
            push_field_le(&mut hasher, &issue.code());
        }
        format!("blake3:{}", hasher.finalize().to_hex())
    }

    pub const fn grants_physical_authority(&self) -> bool { false }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SignedMappedRuntimeContinuity {
    qualification_digest: String,
    report_digest: String,
    policy_digest: String,
    closure_backed_qualification_digest: String,
    continuous_execution_digest: String,
    runtime_policy_digest: String,
    runtime_verifier_ref: String,
    backend_id: String,
    process_instance_id: String,
    runtime_trace_digest: String,
    nix_binding_qualification_digest: String,
    host_identity_digest: String,
    executable_path: String,
    executable_digest: String,
    dependency_closure_digest: String,
    checkpoint_mapping_sequence_digest: String,
    bindings: Vec<SignedMappedCheckpointBinding>,
}

impl SignedMappedRuntimeContinuity {
    pub fn qualification_digest(&self) -> &str { &self.qualification_digest }
    pub fn report_digest(&self) -> &str { &self.report_digest }
    pub fn policy_digest(&self) -> &str { &self.policy_digest }
    pub fn closure_backed_qualification_digest(&self) -> &str {
        &self.closure_backed_qualification_digest
    }
    pub fn continuous_execution_digest(&self) -> &str { &self.continuous_execution_digest }
    pub fn runtime_policy_digest(&self) -> &str { &self.runtime_policy_digest }
    pub fn runtime_verifier_ref(&self) -> &str { &self.runtime_verifier_ref }
    pub fn backend_id(&self) -> &str { &self.backend_id }
    pub fn process_instance_id(&self) -> &str { &self.process_instance_id }
    pub fn runtime_trace_digest(&self) -> &str { &self.runtime_trace_digest }
    pub fn nix_binding_qualification_digest(&self) -> &str {
        &self.nix_binding_qualification_digest
    }
    pub fn host_identity_digest(&self) -> &str { &self.host_identity_digest }
    pub fn executable_path(&self) -> &str { &self.executable_path }
    pub fn executable_digest(&self) -> &str { &self.executable_digest }
    pub fn dependency_closure_digest(&self) -> &str { &self.dependency_closure_digest }
    pub fn checkpoint_mapping_sequence_digest(&self) -> &str {
        &self.checkpoint_mapping_sequence_digest
    }
    pub fn bindings(&self) -> &[SignedMappedCheckpointBinding] { &self.bindings }
    pub const fn every_signed_checkpoint_commits_qualified_mapped_runtime_observation(&self) -> bool {
        true
    }
    pub const fn mapped_runtime_observations_cover_entire_checkpoint_sequence(&self) -> bool {
        true
    }
    pub const fn exact_signed_checkpoint_trace_rebound_to_parent_continuity(&self) -> bool {
        true
    }
    pub const fn observations_bound_to_same_runtime_policy_role_backend_and_host(&self) -> bool {
        true
    }
    pub const fn parent_checkpoint_gap_policy_enforced(&self) -> bool { true }
    pub const fn mapped_runtime_continuity_between_checkpoints_established(&self) -> bool { false }
    pub const fn mapping_continuity_since_exec_established(&self) -> bool { false }
    pub const fn closure_reverified_at_each_checkpoint(&self) -> bool { false }
    pub const fn dependency_closure_atomically_pinned(&self) -> bool { false }
    pub const fn root_resistant_immutability_established(&self) -> bool { false }
    pub const fn trusted_time_established(&self) -> bool { false }
    pub const fn grants_physical_authority(&self) -> bool { false }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SignedMappedRuntimeQualification {
    pub report: SignedMappedRuntimeReport,
    verified: SignedMappedRuntimeContinuity,
}

impl SignedMappedRuntimeQualification {
    pub fn verified(&self) -> &SignedMappedRuntimeContinuity { &self.verified }
    pub fn into_verified(self) -> SignedMappedRuntimeContinuity { self.verified }
}

pub fn bind_signed_mapped_runtime_continuity(
    policy: &SignedMappedRuntimePolicy,
    closure_backed: &ClosureBackedInProcessContinuousVerifierExecution,
    continuous: &ContinuousVerifierExecution,
    evidence: &VerifierRuntimeEvidence,
    observations: &[ObservedMappedNixExecutableRuntime],
) -> Result<SignedMappedRuntimeQualification, SignedMappedRuntimeReport> {
    let policy_digest = policy.canonical_digest();
    let mut report = base_report(policy, policy_digest.clone(), closure_backed);

    if !policy.validate() {
        report.issues.push(SignedMappedRuntimeIssue::InvalidPolicy);
        return Err(finalize(report));
    }
    if !evidence.validate() {
        report.issues.push(SignedMappedRuntimeIssue::InvalidRuntimeEvidence);
        return Err(finalize(report));
    }
    if closure_backed.policy_digest() != policy.expected_closure_backed_policy_digest {
        report.issues.push(SignedMappedRuntimeIssue::ClosureBackedPolicyMismatch);
    }
    if closure_backed.continuous_execution_digest() != continuous.continuity_digest() {
        report.issues.push(SignedMappedRuntimeIssue::ContinuousExecutionMismatch);
    }
    if closure_backed.runtime_policy_digest() != policy.expected_runtime_policy_digest
        || continuous.policy_digest() != policy.expected_runtime_policy_digest
    {
        report.issues.push(SignedMappedRuntimeIssue::RuntimePolicyMismatch);
    }
    if closure_backed.runtime_verifier_ref() != policy.expected_runtime_verifier_ref
        || continuous.verifier_ref() != policy.expected_runtime_verifier_ref
    {
        report.issues.push(SignedMappedRuntimeIssue::RuntimeVerifierMismatch);
    }
    if closure_backed.backend_id() != policy.expected_backend_id {
        report.issues.push(SignedMappedRuntimeIssue::BackendMismatch);
    }
    if closure_backed.process_instance_id() != continuous.process_instance_id() {
        report.issues.push(SignedMappedRuntimeIssue::ProcessInstanceMismatch);
    }
    if closure_backed.runtime_trace_digest() != continuous.runtime_trace_digest() {
        report.issues.push(SignedMappedRuntimeIssue::RuntimeTraceMismatch);
    }
    if closure_backed.input_digest() != continuous.input_digest() {
        report.issues.push(SignedMappedRuntimeIssue::InputDigestMismatch);
    }
    if closure_backed.output_digest() != continuous.output_digest() {
        report.issues.push(SignedMappedRuntimeIssue::OutputDigestMismatch);
    }
    if closure_backed.computation_started_at_ms() != continuous.started_at_ms()
        || closure_backed.computation_completed_at_ms() != continuous.completed_at_ms()
    {
        report.issues.push(SignedMappedRuntimeIssue::ComputationTimeMismatch);
    }
    if closure_backed.continuous_assessed_at_ms() != continuous.assessed_at_ms() {
        report.issues.push(SignedMappedRuntimeIssue::AssessmentTimeMismatch);
    }
    if !report.issues.is_empty() {
        return Err(finalize(report));
    }

    let rebound = match rebind_runtime_evidence(evidence, continuous.assessed_at_ms()) {
        Some(value) => value,
        None => {
            report.issues.push(SignedMappedRuntimeIssue::InvalidRuntimeEvidence);
            return Err(finalize(report));
        }
    };
    if rebound.launch_digest != continuous.launch_attestation_digest() {
        report.issues.push(SignedMappedRuntimeIssue::LaunchDigestMismatch);
    }
    if rebound.final_checkpoint_digest != continuous.final_checkpoint_digest() {
        report.issues.push(SignedMappedRuntimeIssue::FinalCheckpointDigestMismatch);
    }
    if rebound.computation_digest != continuous.computation_attestation_digest() {
        report.issues.push(SignedMappedRuntimeIssue::ComputationDigestMismatch);
    }
    if rebound.runtime_trace_digest != continuous.runtime_trace_digest() {
        report.issues.push(SignedMappedRuntimeIssue::RuntimeTraceMismatch);
    }
    if evidence.launch.process_instance_id != continuous.process_instance_id()
        || evidence.computation.process_instance_id != continuous.process_instance_id()
    {
        report.issues.push(SignedMappedRuntimeIssue::ProcessInstanceMismatch);
    }
    if evidence.launch.verifier_ref != continuous.verifier_ref()
        || evidence.computation.verifier_ref != continuous.verifier_ref()
    {
        report.issues.push(SignedMappedRuntimeIssue::RuntimeVerifierMismatch);
    }
    if evidence.computation.input_digest != continuous.input_digest() {
        report.issues.push(SignedMappedRuntimeIssue::InputDigestMismatch);
    }
    if evidence.computation.output_digest != continuous.output_digest() {
        report.issues.push(SignedMappedRuntimeIssue::OutputDigestMismatch);
    }
    if evidence.computation.started_at_ms != continuous.started_at_ms()
        || evidence.computation.completed_at_ms != continuous.completed_at_ms()
    {
        report.issues.push(SignedMappedRuntimeIssue::ComputationTimeMismatch);
    }
    if !report.issues.is_empty() {
        return Err(finalize(report));
    }

    if evidence.checkpoints.len() != observations.len() || observations.is_empty() {
        report.issues.push(SignedMappedRuntimeIssue::ObservationCountMismatch {
            checkpoints: evidence.checkpoints.len() as u64,
            observations: observations.len() as u64,
        });
        return Err(finalize(report));
    }
    if observations.len() > MAX_CHECKPOINT_BINDINGS {
        report.issues.push(SignedMappedRuntimeIssue::ObservationCountMismatch {
            checkpoints: evidence.checkpoints.len() as u64,
            observations: observations.len() as u64,
        });
        return Err(finalize(report));
    }

    let mut bindings = Vec::with_capacity(observations.len());
    for (index, ((checkpoint, checkpoint_digest), observation)) in evidence
        .checkpoints
        .iter()
        .zip(rebound.checkpoint_digests.iter())
        .zip(observations.iter())
        .enumerate()
    {
        let sequence = index as u64 + 1;
        report.issues.extend(checkpoint_observation_issues(
            policy,
            closure_backed,
            continuous,
            checkpoint.verifier_ref.as_str(),
            checkpoint.process_instance_id.as_str(),
            checkpoint.executable_digest.as_str(),
            checkpoint.dependency_closure_digest.as_str(),
            checkpoint.dynamic_measurement_digest.as_str(),
            checkpoint.observed_at_ms,
            observation,
            sequence,
        ));
        bindings.push(SignedMappedCheckpointBinding {
            sequence,
            checkpoint_digest: checkpoint_digest.clone(),
            checkpoint_dynamic_measurement_digest: checkpoint.dynamic_measurement_digest.clone(),
            observation_qualification_digest: observation.qualification_digest().into(),
            mapped_object_set_digest: observation.mapped_object_set_digest().into(),
            checkpoint_observed_at_ms: checkpoint.observed_at_ms,
            observation_observed_at_ms: observation.observed_at_ms(),
        });
    }
    if !report.issues.is_empty() {
        return Err(finalize(report));
    }

    let sequence_digest = checkpoint_mapping_sequence_digest(&bindings);
    report.checkpoint_mapping_sequence_digest = Some(sequence_digest.clone());
    report.checkpoint_count = bindings.len() as u64;
    report.first_observed_at_ms = bindings
        .first()
        .map(|binding| binding.checkpoint_observed_at_ms)
        .unwrap_or(0);
    report.last_observed_at_ms = bindings
        .last()
        .map(|binding| binding.checkpoint_observed_at_ms)
        .unwrap_or(0);
    report.disposition = SignedMappedRuntimeDisposition::Qualified;

    let report_digest = report.canonical_digest();
    let policy_digest = policy_digest.expect("validated policy has digest");
    let qualification_digest = qualification_digest(
        &policy_digest,
        closure_backed.qualification_digest(),
        continuous.continuity_digest(),
        &sequence_digest,
        &report_digest,
    );
    let verified = SignedMappedRuntimeContinuity {
        qualification_digest,
        report_digest,
        policy_digest,
        closure_backed_qualification_digest: closure_backed.qualification_digest().into(),
        continuous_execution_digest: continuous.continuity_digest().into(),
        runtime_policy_digest: closure_backed.runtime_policy_digest().into(),
        runtime_verifier_ref: closure_backed.runtime_verifier_ref().into(),
        backend_id: closure_backed.backend_id().into(),
        process_instance_id: closure_backed.process_instance_id().into(),
        runtime_trace_digest: closure_backed.runtime_trace_digest().into(),
        nix_binding_qualification_digest: closure_backed.nix_binding_qualification_digest().into(),
        host_identity_digest: closure_backed.host_identity_digest().into(),
        executable_path: closure_backed.executable_path().into(),
        executable_digest: closure_backed.executable_digest().into(),
        dependency_closure_digest: closure_backed.dependency_closure_digest().into(),
        checkpoint_mapping_sequence_digest: sequence_digest,
        bindings,
    };

    Ok(SignedMappedRuntimeQualification { report, verified })
}

#[allow(clippy::too_many_arguments)]
fn checkpoint_observation_issues(
    policy: &SignedMappedRuntimePolicy,
    parent: &ClosureBackedInProcessContinuousVerifierExecution,
    continuous: &ContinuousVerifierExecution,
    checkpoint_verifier_ref: &str,
    checkpoint_process_instance_id: &str,
    checkpoint_executable_digest: &str,
    checkpoint_closure_digest: &str,
    checkpoint_dynamic_measurement_digest: &str,
    checkpoint_observed_at_ms: u64,
    observation: &ObservedMappedNixExecutableRuntime,
    sequence: u64,
) -> Vec<SignedMappedRuntimeIssue> {
    let mut issues = Vec::new();
    if observation.policy_digest() != policy.expected_mapped_runtime_policy_digest {
        issues.push(SignedMappedRuntimeIssue::MappedRuntimePolicyMismatch { sequence });
    }
    if observation.nix_binding_qualification_digest() != parent.nix_binding_qualification_digest() {
        issues.push(SignedMappedRuntimeIssue::NixBindingQualificationMismatch { sequence });
    }
    if observation.runtime_policy_digest() != parent.runtime_policy_digest() {
        issues.push(SignedMappedRuntimeIssue::ObservationRuntimePolicyMismatch { sequence });
    }
    if observation.runtime_verifier_ref() != parent.runtime_verifier_ref() {
        issues.push(SignedMappedRuntimeIssue::ObservationVerifierMismatch { sequence });
    }
    if observation.backend_id() != parent.backend_id() {
        issues.push(SignedMappedRuntimeIssue::ObservationBackendMismatch { sequence });
    }
    if observation.host_identity_digest() != parent.host_identity_digest() {
        issues.push(SignedMappedRuntimeIssue::ObservationHostIdentityMismatch { sequence });
    }
    if observation.host_executable_path() != parent.executable_path() {
        issues.push(SignedMappedRuntimeIssue::ObservationExecutablePathMismatch { sequence });
    }
    if observation.host_executable_digest() != parent.executable_digest() {
        issues.push(SignedMappedRuntimeIssue::ObservationExecutableDigestMismatch { sequence });
    }
    if observation.closure_digest() != parent.dependency_closure_digest() {
        issues.push(SignedMappedRuntimeIssue::ObservationClosureDigestMismatch { sequence });
    }
    if observation.observed_at_ms() != checkpoint_observed_at_ms {
        issues.push(SignedMappedRuntimeIssue::ObservationTimeMismatch { sequence });
    }
    if checkpoint_verifier_ref != continuous.verifier_ref() {
        issues.push(SignedMappedRuntimeIssue::CheckpointVerifierMismatch { sequence });
    }
    if checkpoint_process_instance_id != continuous.process_instance_id() {
        issues.push(SignedMappedRuntimeIssue::CheckpointProcessMismatch { sequence });
    }
    if checkpoint_executable_digest != parent.executable_digest() {
        issues.push(SignedMappedRuntimeIssue::CheckpointExecutableDigestMismatch { sequence });
    }
    if checkpoint_closure_digest != parent.dependency_closure_digest() {
        issues.push(SignedMappedRuntimeIssue::CheckpointClosureDigestMismatch { sequence });
    }
    if checkpoint_dynamic_measurement_digest != observation.qualification_digest() {
        issues.push(SignedMappedRuntimeIssue::DynamicMeasurementDigestMismatch { sequence });
    }
    issues
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ReboundRuntimeEvidence {
    launch_digest: String,
    checkpoint_digests: Vec<String>,
    final_checkpoint_digest: String,
    computation_digest: String,
    runtime_trace_digest: String,
}

fn rebind_runtime_evidence(
    evidence: &VerifierRuntimeEvidence,
    assessed_at_ms: u64,
) -> Option<ReboundRuntimeEvidence> {
    if !evidence.validate() {
        return None;
    }
    let policy_digest = evidence.launch.policy_digest.clone();
    let launch_digest = evidence.launch.canonical_digest()?;
    let checkpoint_digests = evidence
        .checkpoints
        .iter()
        .map(|checkpoint| checkpoint.canonical_digest())
        .collect::<Option<Vec<_>>>()?;
    let final_checkpoint_digest = checkpoint_digests.last()?.clone();
    let computation_digest = evidence.computation.canonical_digest()?;
    let runtime_trace_digest = runtime_trace_digest(
        &policy_digest,
        &launch_digest,
        &checkpoint_digests,
        &computation_digest,
        assessed_at_ms,
    );
    Some(ReboundRuntimeEvidence {
        launch_digest,
        checkpoint_digests,
        final_checkpoint_digest,
        computation_digest,
        runtime_trace_digest,
    })
}

fn runtime_trace_digest(
    policy_digest: &str,
    launch_digest: &str,
    checkpoints: &[String],
    computation_digest: &str,
    assessed_at_ms: u64,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(RUNTIME_TRACE_DIGEST_DOMAIN);
    push_field_be(&mut hasher, policy_digest);
    push_field_be(&mut hasher, launch_digest);
    for checkpoint in checkpoints {
        push_field_be(&mut hasher, checkpoint);
    }
    push_field_be(&mut hasher, computation_digest);
    hasher.update(&assessed_at_ms.to_be_bytes());
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn checkpoint_mapping_sequence_digest(bindings: &[SignedMappedCheckpointBinding]) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(CHECKPOINT_MAPPING_SEQUENCE_DIGEST_DOMAIN);
    hasher.update(&(bindings.len() as u64).to_le_bytes());
    for binding in bindings {
        hasher.update(&binding.sequence.to_le_bytes());
        push_field_le(&mut hasher, &binding.checkpoint_digest);
        push_field_le(&mut hasher, &binding.checkpoint_dynamic_measurement_digest);
        push_field_le(&mut hasher, &binding.observation_qualification_digest);
        push_field_le(&mut hasher, &binding.mapped_object_set_digest);
        hasher.update(&binding.checkpoint_observed_at_ms.to_le_bytes());
        hasher.update(&binding.observation_observed_at_ms.to_le_bytes());
    }
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn qualification_digest(
    policy_digest: &str,
    closure_backed_qualification_digest: &str,
    continuous_execution_digest: &str,
    checkpoint_mapping_sequence_digest: &str,
    report_digest: &str,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(QUALIFICATION_DIGEST_DOMAIN);
    for field in [
        policy_digest,
        closure_backed_qualification_digest,
        continuous_execution_digest,
        checkpoint_mapping_sequence_digest,
        report_digest,
    ] {
        push_field_le(&mut hasher, field);
    }
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn base_report(
    policy: &SignedMappedRuntimePolicy,
    policy_digest: Option<String>,
    parent: &ClosureBackedInProcessContinuousVerifierExecution,
) -> SignedMappedRuntimeReport {
    SignedMappedRuntimeReport {
        schema_version: SIGNED_MAPPED_RUNTIME_REPORT_SCHEMA_V1.into(),
        policy_id: policy.policy_id.clone(),
        policy_digest,
        closure_backed_policy_digest: parent.policy_digest().into(),
        closure_backed_qualification_digest: parent.qualification_digest().into(),
        continuous_execution_digest: parent.continuous_execution_digest().into(),
        runtime_policy_digest: parent.runtime_policy_digest().into(),
        runtime_verifier_ref: parent.runtime_verifier_ref().into(),
        backend_id: parent.backend_id().into(),
        process_instance_id: parent.process_instance_id().into(),
        runtime_trace_digest: parent.runtime_trace_digest().into(),
        nix_binding_qualification_digest: parent.nix_binding_qualification_digest().into(),
        host_identity_digest: parent.host_identity_digest().into(),
        executable_path: parent.executable_path().into(),
        executable_digest: parent.executable_digest().into(),
        dependency_closure_digest: parent.dependency_closure_digest().into(),
        checkpoint_mapping_sequence_digest: None,
        checkpoint_count: 0,
        first_observed_at_ms: 0,
        last_observed_at_ms: 0,
        disposition: SignedMappedRuntimeDisposition::Invalid,
        issues: Vec::new(),
    }
}

fn finalize(mut report: SignedMappedRuntimeReport) -> SignedMappedRuntimeReport {
    report.disposition = if report.issues.iter().any(SignedMappedRuntimeIssue::is_invalid) {
        SignedMappedRuntimeDisposition::Invalid
    } else {
        SignedMappedRuntimeDisposition::Blocked
    };
    report
}

fn canonical_text(value: &str) -> bool {
    !value.is_empty()
        && value == value.trim()
        && value.len() <= MAX_TEXT_BYTES
        && !value.chars().any(char::is_control)
}

fn valid_blake3_digest(value: &str) -> bool {
    let Some(hex) = value.strip_prefix("blake3:") else { return false; };
    hex.len() == 64
        && hex
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn valid_refs(refs: &[String]) -> bool {
    refs.len() <= MAX_EVIDENCE_REFS
        && refs.iter().all(|value| canonical_text(value))
        && unique(refs)
}

fn unique(values: &[String]) -> bool {
    let mut seen = BTreeSet::new();
    values.iter().all(|value| seen.insert(value.as_str()))
}

fn push_field_be(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_be_bytes());
    hasher.update(value.as_bytes());
}

fn push_field_le(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn push_sorted_refs_le(hasher: &mut blake3::Hasher, refs: &[String]) {
    let mut refs = refs.to_vec();
    refs.sort();
    hasher.update(&(refs.len() as u64).to_le_bytes());
    for reference in refs {
        push_field_le(hasher, &reference);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(label: &str) -> String {
        format!("blake3:{}", blake3::hash(label.as_bytes()).to_hex())
    }

    fn policy() -> SignedMappedRuntimePolicy {
        SignedMappedRuntimePolicy {
            schema_version: SIGNED_MAPPED_RUNTIME_POLICY_SCHEMA_V1.into(),
            policy_id: "policy:signed-mapped-runtime:1".into(),
            expected_closure_backed_policy_digest: d("parent-policy"),
            expected_mapped_runtime_policy_digest: d("mapped-policy"),
            expected_runtime_policy_digest: d("runtime-policy"),
            expected_runtime_verifier_ref: "verifier:host".into(),
            expected_backend_id: "backend:in-process".into(),
            evidence_refs: vec!["review:a".into(), "review:b".into()],
        }
    }

    fn binding(sequence: u64, suffix: &str) -> SignedMappedCheckpointBinding {
        SignedMappedCheckpointBinding {
            sequence,
            checkpoint_digest: d(&format!("checkpoint:{suffix}")),
            checkpoint_dynamic_measurement_digest: d(&format!("observation:{suffix}")),
            observation_qualification_digest: d(&format!("observation:{suffix}")),
            mapped_object_set_digest: d(&format!("mapped-set:{suffix}")),
            checkpoint_observed_at_ms: 1_000 + sequence,
            observation_observed_at_ms: 1_000 + sequence,
        }
    }

    #[test]
    fn policy_reference_order_is_nonsemantic() {
        let left = policy();
        let mut right = left.clone();
        right.evidence_refs.reverse();
        assert_eq!(left.canonical_digest(), right.canonical_digest());
        right.expected_backend_id = "backend:other".into();
        assert_ne!(left.canonical_digest(), right.canonical_digest());
    }

    #[test]
    fn checkpoint_mapping_sequence_digest_binds_order_and_observation() {
        let left = vec![binding(1, "a"), binding(2, "b")];
        let reversed = vec![binding(2, "b"), binding(1, "a")];
        assert_ne!(
            checkpoint_mapping_sequence_digest(&left),
            checkpoint_mapping_sequence_digest(&reversed)
        );
        let mut changed = left.clone();
        changed[1].mapped_object_set_digest = d("changed");
        assert_ne!(
            checkpoint_mapping_sequence_digest(&left),
            checkpoint_mapping_sequence_digest(&changed)
        );
    }

    #[test]
    fn runtime_trace_digest_uses_parent_big_endian_encoding() {
        let policy = d("policy");
        let launch = d("launch");
        let checkpoints = vec![d("checkpoint:1"), d("checkpoint:2")];
        let computation = d("computation");
        let assessed_at_ms = 42u64;

        let digest = runtime_trace_digest(
            &policy,
            &launch,
            &checkpoints,
            &computation,
            assessed_at_ms,
        );

        let mut oracle = blake3::Hasher::new();
        oracle.update(RUNTIME_TRACE_DIGEST_DOMAIN);
        for value in [&policy, &launch, &checkpoints[0], &checkpoints[1], &computation] {
            oracle.update(&(value.len() as u64).to_be_bytes());
            oracle.update(value.as_bytes());
        }
        oracle.update(&assessed_at_ms.to_be_bytes());
        assert_eq!(digest, format!("blake3:{}", oracle.finalize().to_hex()));
    }
}
