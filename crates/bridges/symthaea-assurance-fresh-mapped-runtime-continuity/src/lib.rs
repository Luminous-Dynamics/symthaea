// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Challenge-bound live mapped-runtime observations committed by signed checkpoints.
//!
//! This bridge closes one replay gap left intentionally open by the first
//! signed mapped-runtime composition. A checkpoint challenge is validated first;
//! the live mapped-runtime observer is then invoked inside this constructor; and
//! only the resulting opaque fresh-observation capability can be committed by a
//! checkpoint's signed `dynamic_measurement_digest`.
//!
//! The theorem is deliberately local to one already-qualified runtime trace.
//! It does not establish trusted challenge authority, unpredictable entropy,
//! global replay resistance, continuous mapping identity between checkpoints,
//! launch-time loader atomicity, trusted time, or physical authority.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use symthaea_assurance_closure_backed_in_process_continuous::
    ClosureBackedInProcessContinuousVerifierExecution;
use symthaea_assurance_nix_bound_in_process_runtime::NixBoundInProcessRuntimePolicy;
use symthaea_assurance_nix_runtime_closure::NixRuntimeClosureQualification;
use symthaea_assurance_observed_mapped_nix_executable_runtime::{
    MappedExecutableRuntimePolicy, MappedExecutableRuntimeQualification,
    ObservedMappedNixExecutableRuntime,
};
#[cfg(target_os = "linux")]
use symthaea_assurance_observed_mapped_nix_executable_runtime::
    observe_mapped_nix_executable_runtime;
use symthaea_evidence_verifier_runtime_continuity::{
    ContinuousVerifierExecution, VerifierRuntimeEvidence,
};

pub const FRESH_MAPPED_RUNTIME_POLICY_SCHEMA_V1: &str =
    "symthaea.assurance.fresh-mapped-runtime-continuity-policy.v1";
pub const MAPPED_RUNTIME_CHECKPOINT_CHALLENGE_SCHEMA_V1: &str =
    "symthaea.assurance.mapped-runtime-checkpoint-challenge.v1";
pub const FRESH_MAPPED_RUNTIME_OBSERVATION_REPORT_SCHEMA_V1: &str =
    "symthaea.assurance.fresh-mapped-runtime-observation-report.v1";
pub const FRESH_MAPPED_RUNTIME_CONTINUITY_REPORT_SCHEMA_V1: &str =
    "symthaea.assurance.fresh-mapped-runtime-continuity-report.v1";

const POLICY_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.fresh-mapped-runtime-continuity-policy.digest.v1\0";
const CHALLENGE_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.mapped-runtime-checkpoint-challenge.digest.v1\0";
const OBSERVATION_REPORT_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.fresh-mapped-runtime-observation-report.digest.v1\0";
const OBSERVATION_QUALIFICATION_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.fresh-mapped-runtime-observation-qualification.digest.v1\0";
const RUNTIME_TRACE_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.verifier-runtime-trace.digest.v1\0";
const FRESH_SEQUENCE_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.fresh-mapped-runtime-checkpoint-sequence.digest.v1\0";
const CONTINUITY_REPORT_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.fresh-mapped-runtime-continuity-report.digest.v1\0";
const CONTINUITY_QUALIFICATION_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.fresh-mapped-runtime-continuity-qualification.digest.v1\0";
const MAX_TEXT_BYTES: usize = 4096;
const MAX_EVIDENCE_REFS: usize = 128;
const MAX_CHECKPOINT_BINDINGS: usize = 65_536;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FreshMappedRuntimePolicy {
    pub schema_version: String,
    pub policy_id: String,
    pub expected_closure_backed_policy_digest: String,
    pub expected_nix_binding_policy_digest: String,
    pub expected_mapped_runtime_policy_digest: String,
    pub expected_runtime_policy_digest: String,
    pub expected_runtime_verifier_ref: String,
    pub expected_backend_id: String,
    pub evidence_refs: Vec<String>,
}

impl FreshMappedRuntimePolicy {
    pub fn validate(&self) -> bool {
        self.schema_version == FRESH_MAPPED_RUNTIME_POLICY_SCHEMA_V1
            && canonical_text(&self.policy_id)
            && valid_blake3_digest(&self.expected_closure_backed_policy_digest)
            && valid_blake3_digest(&self.expected_nix_binding_policy_digest)
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
            self.expected_nix_binding_policy_digest.as_str(),
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

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MappedRuntimeCheckpointChallenge {
    pub schema_version: String,
    pub process_instance_id: String,
    pub runtime_policy_digest: String,
    pub verifier_ref: String,
    pub checkpoint_sequence: u64,
    pub checkpoint_monotonic_counter: u64,
    pub previous_checkpoint_digest: Option<String>,
    pub nonce_blake3_hex: String,
}

impl MappedRuntimeCheckpointChallenge {
    pub fn validate(&self) -> bool {
        self.schema_version == MAPPED_RUNTIME_CHECKPOINT_CHALLENGE_SCHEMA_V1
            && canonical_text(&self.process_instance_id)
            && valid_blake3_digest(&self.runtime_policy_digest)
            && canonical_text(&self.verifier_ref)
            && self.checkpoint_sequence > 0
            && self.checkpoint_monotonic_counter > 0
            && match self.checkpoint_sequence {
                1 => self.previous_checkpoint_digest.is_none(),
                _ => self
                    .previous_checkpoint_digest
                    .as_deref()
                    .is_some_and(valid_blake3_digest),
            }
            && lower_hex_exact(&self.nonce_blake3_hex, 64)
    }

    pub fn canonical_digest(&self) -> Option<String> {
        if !self.validate() {
            return None;
        }
        let mut hasher = blake3::Hasher::new();
        hasher.update(CHALLENGE_DIGEST_DOMAIN);
        for field in [
            self.schema_version.as_str(),
            self.process_instance_id.as_str(),
            self.runtime_policy_digest.as_str(),
            self.verifier_ref.as_str(),
        ] {
            push_field_le(&mut hasher, field);
        }
        hasher.update(&self.checkpoint_sequence.to_le_bytes());
        hasher.update(&self.checkpoint_monotonic_counter.to_le_bytes());
        push_optional_field_le(&mut hasher, self.previous_checkpoint_digest.as_deref());
        push_field_le(&mut hasher, &self.nonce_blake3_hex);
        Some(format!("blake3:{}", hasher.finalize().to_hex()))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FreshMappedObservationDisposition {
    Invalid,
    Blocked,
    Qualified,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FreshMappedObservationIssue {
    InvalidPolicy,
    InvalidChallenge,
    ClosureBackedPolicyMismatch,
    NixBindingPolicyMismatch,
    MappedRuntimePolicyMismatch,
    RuntimePolicyMismatch,
    RuntimeVerifierMismatch,
    BackendMismatch,
    ProcessInstanceMismatch,
    ParentNixBindingMismatch,
    BoundRuntimePolicyMismatch,
    BoundVerifierMismatch,
    BoundBackendMismatch,
    BoundHostIdentityMismatch,
    BoundExecutablePathMismatch,
    BoundExecutableDigestMismatch,
    BoundClosureDigestMismatch,
    LiveObservationUnavailable(String),
    ObservationPolicyMismatch,
    ObservationNixBindingMismatch,
    ObservationRuntimePolicyMismatch,
    ObservationVerifierMismatch,
    ObservationBackendMismatch,
    ObservationHostIdentityMismatch,
    ObservationExecutablePathMismatch,
    ObservationExecutableDigestMismatch,
    ObservationClosureDigestMismatch,
}

impl FreshMappedObservationIssue {
    fn is_invalid(&self) -> bool {
        matches!(self, Self::InvalidPolicy | Self::InvalidChallenge)
    }

    fn code(&self) -> String {
        match self {
            Self::InvalidPolicy => "invalid-policy".into(),
            Self::InvalidChallenge => "invalid-challenge".into(),
            Self::ClosureBackedPolicyMismatch => "closure-backed-policy-mismatch".into(),
            Self::NixBindingPolicyMismatch => "nix-binding-policy-mismatch".into(),
            Self::MappedRuntimePolicyMismatch => "mapped-runtime-policy-mismatch".into(),
            Self::RuntimePolicyMismatch => "runtime-policy-mismatch".into(),
            Self::RuntimeVerifierMismatch => "runtime-verifier-mismatch".into(),
            Self::BackendMismatch => "backend-mismatch".into(),
            Self::ProcessInstanceMismatch => "process-instance-mismatch".into(),
            Self::ParentNixBindingMismatch => "parent-nix-binding-mismatch".into(),
            Self::BoundRuntimePolicyMismatch => "bound-runtime-policy-mismatch".into(),
            Self::BoundVerifierMismatch => "bound-verifier-mismatch".into(),
            Self::BoundBackendMismatch => "bound-backend-mismatch".into(),
            Self::BoundHostIdentityMismatch => "bound-host-identity-mismatch".into(),
            Self::BoundExecutablePathMismatch => "bound-executable-path-mismatch".into(),
            Self::BoundExecutableDigestMismatch => "bound-executable-digest-mismatch".into(),
            Self::BoundClosureDigestMismatch => "bound-closure-digest-mismatch".into(),
            Self::LiveObservationUnavailable(reason) => {
                format!("live-observation-unavailable:{reason}")
            }
            Self::ObservationPolicyMismatch => "observation-policy-mismatch".into(),
            Self::ObservationNixBindingMismatch => "observation-nix-binding-mismatch".into(),
            Self::ObservationRuntimePolicyMismatch => "observation-runtime-policy-mismatch".into(),
            Self::ObservationVerifierMismatch => "observation-verifier-mismatch".into(),
            Self::ObservationBackendMismatch => "observation-backend-mismatch".into(),
            Self::ObservationHostIdentityMismatch => "observation-host-identity-mismatch".into(),
            Self::ObservationExecutablePathMismatch => "observation-executable-path-mismatch".into(),
            Self::ObservationExecutableDigestMismatch => "observation-executable-digest-mismatch".into(),
            Self::ObservationClosureDigestMismatch => "observation-closure-digest-mismatch".into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FreshMappedObservationReport {
    pub schema_version: String,
    pub policy_id: String,
    pub policy_digest: Option<String>,
    pub challenge_digest: Option<String>,
    pub challenge_nonce_blake3_hex: String,
    pub process_instance_id: String,
    pub checkpoint_sequence: u64,
    pub checkpoint_monotonic_counter: u64,
    pub previous_checkpoint_digest: Option<String>,
    pub closure_backed_qualification_digest: String,
    pub nix_binding_qualification_digest: String,
    pub mapped_runtime_policy_digest: String,
    pub raw_observation_qualification_digest: Option<String>,
    pub mapped_object_set_digest: Option<String>,
    pub runtime_policy_digest: String,
    pub runtime_verifier_ref: String,
    pub backend_id: String,
    pub host_identity_digest: String,
    pub executable_path: String,
    pub executable_digest: String,
    pub dependency_closure_digest: String,
    pub observed_at_ms: u64,
    pub disposition: FreshMappedObservationDisposition,
    pub issues: Vec<FreshMappedObservationIssue>,
}

impl FreshMappedObservationReport {
    pub fn canonical_digest(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(OBSERVATION_REPORT_DIGEST_DOMAIN);
        for field in [
            self.schema_version.as_str(),
            self.policy_id.as_str(),
            self.policy_digest.as_deref().unwrap_or("-"),
            self.challenge_digest.as_deref().unwrap_or("-"),
            self.challenge_nonce_blake3_hex.as_str(),
            self.process_instance_id.as_str(),
            self.closure_backed_qualification_digest.as_str(),
            self.nix_binding_qualification_digest.as_str(),
            self.mapped_runtime_policy_digest.as_str(),
            self.raw_observation_qualification_digest.as_deref().unwrap_or("-"),
            self.mapped_object_set_digest.as_deref().unwrap_or("-"),
            self.runtime_policy_digest.as_str(),
            self.runtime_verifier_ref.as_str(),
            self.backend_id.as_str(),
            self.host_identity_digest.as_str(),
            self.executable_path.as_str(),
            self.executable_digest.as_str(),
            self.dependency_closure_digest.as_str(),
        ] {
            push_field_le(&mut hasher, field);
        }
        hasher.update(&self.checkpoint_sequence.to_le_bytes());
        hasher.update(&self.checkpoint_monotonic_counter.to_le_bytes());
        push_optional_field_le(&mut hasher, self.previous_checkpoint_digest.as_deref());
        hasher.update(&self.observed_at_ms.to_le_bytes());
        push_field_le(
            &mut hasher,
            match self.disposition {
                FreshMappedObservationDisposition::Invalid => "invalid",
                FreshMappedObservationDisposition::Blocked => "blocked",
                FreshMappedObservationDisposition::Qualified => "qualified",
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
pub struct FreshMappedRuntimeObservation {
    qualification_digest: String,
    report_digest: String,
    policy_digest: String,
    challenge_digest: String,
    challenge_nonce_blake3_hex: String,
    process_instance_id: String,
    checkpoint_sequence: u64,
    checkpoint_monotonic_counter: u64,
    previous_checkpoint_digest: Option<String>,
    closure_backed_qualification_digest: String,
    nix_binding_qualification_digest: String,
    mapped_runtime_policy_digest: String,
    raw_observation_qualification_digest: String,
    mapped_object_set_digest: String,
    runtime_policy_digest: String,
    runtime_verifier_ref: String,
    backend_id: String,
    host_identity_digest: String,
    executable_path: String,
    executable_digest: String,
    dependency_closure_digest: String,
    observed_at_ms: u64,
}

impl FreshMappedRuntimeObservation {
    pub fn qualification_digest(&self) -> &str { &self.qualification_digest }
    pub fn report_digest(&self) -> &str { &self.report_digest }
    pub fn policy_digest(&self) -> &str { &self.policy_digest }
    pub fn challenge_digest(&self) -> &str { &self.challenge_digest }
    pub fn challenge_nonce_blake3_hex(&self) -> &str { &self.challenge_nonce_blake3_hex }
    pub fn process_instance_id(&self) -> &str { &self.process_instance_id }
    pub const fn checkpoint_sequence(&self) -> u64 { self.checkpoint_sequence }
    pub const fn checkpoint_monotonic_counter(&self) -> u64 { self.checkpoint_monotonic_counter }
    pub fn previous_checkpoint_digest(&self) -> Option<&str> {
        self.previous_checkpoint_digest.as_deref()
    }
    pub fn closure_backed_qualification_digest(&self) -> &str {
        &self.closure_backed_qualification_digest
    }
    pub fn nix_binding_qualification_digest(&self) -> &str {
        &self.nix_binding_qualification_digest
    }
    pub fn mapped_runtime_policy_digest(&self) -> &str { &self.mapped_runtime_policy_digest }
    pub fn raw_observation_qualification_digest(&self) -> &str {
        &self.raw_observation_qualification_digest
    }
    pub fn mapped_object_set_digest(&self) -> &str { &self.mapped_object_set_digest }
    pub fn runtime_policy_digest(&self) -> &str { &self.runtime_policy_digest }
    pub fn runtime_verifier_ref(&self) -> &str { &self.runtime_verifier_ref }
    pub fn backend_id(&self) -> &str { &self.backend_id }
    pub fn host_identity_digest(&self) -> &str { &self.host_identity_digest }
    pub fn executable_path(&self) -> &str { &self.executable_path }
    pub fn executable_digest(&self) -> &str { &self.executable_digest }
    pub fn dependency_closure_digest(&self) -> &str { &self.dependency_closure_digest }
    pub const fn observed_at_ms(&self) -> u64 { self.observed_at_ms }
    pub const fn constructor_requires_challenge_before_live_observation(&self) -> bool { true }
    pub const fn challenge_coordinates_committed_into_qualification(&self) -> bool { true }
    pub const fn live_mapped_runtime_observation_performed_by_constructor(&self) -> bool { true }
    pub const fn trusted_challenge_authority_established(&self) -> bool { false }
    pub const fn challenge_nonce_unpredictability_established(&self) -> bool { false }
    pub const fn trusted_time_established(&self) -> bool { false }
    pub const fn grants_physical_authority(&self) -> bool { false }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FreshMappedRuntimeObservationQualification {
    pub report: FreshMappedObservationReport,
    pub raw: MappedExecutableRuntimeQualification,
    fresh: FreshMappedRuntimeObservation,
}

impl FreshMappedRuntimeObservationQualification {
    pub fn fresh(&self) -> &FreshMappedRuntimeObservation { &self.fresh }
    pub fn into_fresh(self) -> FreshMappedRuntimeObservation { self.fresh }
}

#[cfg(target_os = "linux")]
pub fn observe_fresh_mapped_runtime_for_checkpoint(
    policy: &FreshMappedRuntimePolicy,
    mapped_policy: &MappedExecutableRuntimePolicy,
    challenge: &MappedRuntimeCheckpointChallenge,
    parent: &ClosureBackedInProcessContinuousVerifierExecution,
    bound: &NixBoundInProcessRuntimePolicy,
    closure: &NixRuntimeClosureQualification,
    observed_at_ms: u64,
) -> Result<FreshMappedRuntimeObservationQualification, FreshMappedObservationReport> {
    let policy_digest = policy.canonical_digest();
    let challenge_digest = challenge.canonical_digest();
    let mapped_policy_digest = mapped_policy.canonical_digest();
    let mut report = fresh_base_report(
        policy,
        policy_digest.clone(),
        challenge,
        challenge_digest.clone(),
        parent,
        mapped_policy_digest.as_deref().unwrap_or("-"),
        observed_at_ms,
    );

    if !policy.validate() {
        report.issues.push(FreshMappedObservationIssue::InvalidPolicy);
        return Err(finalize_fresh(report));
    }
    if !challenge.validate() || challenge_digest.is_none() {
        report.issues.push(FreshMappedObservationIssue::InvalidChallenge);
        return Err(finalize_fresh(report));
    }
    if parent.policy_digest() != policy.expected_closure_backed_policy_digest {
        report.issues.push(FreshMappedObservationIssue::ClosureBackedPolicyMismatch);
    }
    if bound.policy_digest() != policy.expected_nix_binding_policy_digest {
        report.issues.push(FreshMappedObservationIssue::NixBindingPolicyMismatch);
    }
    if mapped_policy_digest.as_deref() != Some(policy.expected_mapped_runtime_policy_digest.as_str()) {
        report.issues.push(FreshMappedObservationIssue::MappedRuntimePolicyMismatch);
    }
    if parent.runtime_policy_digest() != policy.expected_runtime_policy_digest
        || challenge.runtime_policy_digest != policy.expected_runtime_policy_digest
    {
        report.issues.push(FreshMappedObservationIssue::RuntimePolicyMismatch);
    }
    if parent.runtime_verifier_ref() != policy.expected_runtime_verifier_ref
        || challenge.verifier_ref != policy.expected_runtime_verifier_ref
    {
        report.issues.push(FreshMappedObservationIssue::RuntimeVerifierMismatch);
    }
    if parent.backend_id() != policy.expected_backend_id {
        report.issues.push(FreshMappedObservationIssue::BackendMismatch);
    }
    if challenge.process_instance_id != parent.process_instance_id() {
        report.issues.push(FreshMappedObservationIssue::ProcessInstanceMismatch);
    }
    if bound.qualification_digest() != parent.nix_binding_qualification_digest() {
        report.issues.push(FreshMappedObservationIssue::ParentNixBindingMismatch);
    }
    if bound.runtime_policy_digest() != parent.runtime_policy_digest() {
        report.issues.push(FreshMappedObservationIssue::BoundRuntimePolicyMismatch);
    }
    if bound.runtime_verifier_ref() != parent.runtime_verifier_ref() {
        report.issues.push(FreshMappedObservationIssue::BoundVerifierMismatch);
    }
    if bound.backend_id() != parent.backend_id() {
        report.issues.push(FreshMappedObservationIssue::BoundBackendMismatch);
    }
    if bound.host_identity_digest() != parent.host_identity_digest() {
        report.issues.push(FreshMappedObservationIssue::BoundHostIdentityMismatch);
    }
    if bound.executable_path() != parent.executable_path() {
        report.issues.push(FreshMappedObservationIssue::BoundExecutablePathMismatch);
    }
    if bound.executable_digest() != parent.executable_digest() {
        report.issues.push(FreshMappedObservationIssue::BoundExecutableDigestMismatch);
    }
    if bound.dependency_closure_digest() != parent.dependency_closure_digest() {
        report.issues.push(FreshMappedObservationIssue::BoundClosureDigestMismatch);
    }
    if !report.issues.is_empty() {
        return Err(finalize_fresh(report));
    }

    // The challenge has been structurally and semantically validated before the
    // live observer is invoked. Callers cannot supply an already-created raw
    // observation to this constructor.
    let raw = match observe_mapped_nix_executable_runtime(
        mapped_policy,
        bound,
        closure,
        observed_at_ms,
    ) {
        Ok(value) => value,
        Err(raw_report) => {
            report.issues.push(FreshMappedObservationIssue::LiveObservationUnavailable(
                raw_report.canonical_digest(),
            ));
            return Err(finalize_fresh(report));
        }
    };
    let observation = raw.observed();
    report.raw_observation_qualification_digest = Some(observation.qualification_digest().into());
    report.mapped_object_set_digest = Some(observation.mapped_object_set_digest().into());

    report.issues.extend(observation_identity_issues(policy, parent, observation));
    if !report.issues.is_empty() {
        return Err(finalize_fresh(report));
    }

    report.disposition = FreshMappedObservationDisposition::Qualified;
    let report_digest = report.canonical_digest();
    let policy_digest = policy_digest.expect("validated policy has digest");
    let challenge_digest = challenge_digest.expect("validated challenge has digest");
    let qualification_digest = fresh_observation_qualification_digest(
        &policy_digest,
        &challenge_digest,
        parent.qualification_digest(),
        observation.qualification_digest(),
        observation.mapped_object_set_digest(),
        &report_digest,
    );
    let fresh = FreshMappedRuntimeObservation {
        qualification_digest,
        report_digest,
        policy_digest,
        challenge_digest,
        challenge_nonce_blake3_hex: challenge.nonce_blake3_hex.clone(),
        process_instance_id: challenge.process_instance_id.clone(),
        checkpoint_sequence: challenge.checkpoint_sequence,
        checkpoint_monotonic_counter: challenge.checkpoint_monotonic_counter,
        previous_checkpoint_digest: challenge.previous_checkpoint_digest.clone(),
        closure_backed_qualification_digest: parent.qualification_digest().into(),
        nix_binding_qualification_digest: parent.nix_binding_qualification_digest().into(),
        mapped_runtime_policy_digest: observation.policy_digest().into(),
        raw_observation_qualification_digest: observation.qualification_digest().into(),
        mapped_object_set_digest: observation.mapped_object_set_digest().into(),
        runtime_policy_digest: parent.runtime_policy_digest().into(),
        runtime_verifier_ref: parent.runtime_verifier_ref().into(),
        backend_id: parent.backend_id().into(),
        host_identity_digest: parent.host_identity_digest().into(),
        executable_path: parent.executable_path().into(),
        executable_digest: parent.executable_digest().into(),
        dependency_closure_digest: parent.dependency_closure_digest().into(),
        observed_at_ms: observation.observed_at_ms(),
    };

    Ok(FreshMappedRuntimeObservationQualification { report, raw, fresh })
}

#[cfg(not(target_os = "linux"))]
pub fn observe_fresh_mapped_runtime_for_checkpoint(
    policy: &FreshMappedRuntimePolicy,
    mapped_policy: &MappedExecutableRuntimePolicy,
    challenge: &MappedRuntimeCheckpointChallenge,
    parent: &ClosureBackedInProcessContinuousVerifierExecution,
    _bound: &NixBoundInProcessRuntimePolicy,
    _closure: &NixRuntimeClosureQualification,
    observed_at_ms: u64,
) -> Result<FreshMappedRuntimeObservationQualification, FreshMappedObservationReport> {
    let policy_digest = policy.canonical_digest();
    let challenge_digest = challenge.canonical_digest();
    let mapped_policy_digest = mapped_policy.canonical_digest();
    let mut report = fresh_base_report(
        policy,
        policy_digest,
        challenge,
        challenge_digest,
        parent,
        mapped_policy_digest.as_deref().unwrap_or("-"),
        observed_at_ms,
    );
    report.issues.push(FreshMappedObservationIssue::LiveObservationUnavailable(
        "Linux /proc/self/maps and /proc/self/mem are required".into(),
    ));
    Err(finalize_fresh(report))
}

fn observation_identity_issues(
    policy: &FreshMappedRuntimePolicy,
    parent: &ClosureBackedInProcessContinuousVerifierExecution,
    observation: &ObservedMappedNixExecutableRuntime,
) -> Vec<FreshMappedObservationIssue> {
    let mut issues = Vec::new();
    if observation.policy_digest() != policy.expected_mapped_runtime_policy_digest {
        issues.push(FreshMappedObservationIssue::ObservationPolicyMismatch);
    }
    if observation.nix_binding_qualification_digest() != parent.nix_binding_qualification_digest() {
        issues.push(FreshMappedObservationIssue::ObservationNixBindingMismatch);
    }
    if observation.runtime_policy_digest() != parent.runtime_policy_digest() {
        issues.push(FreshMappedObservationIssue::ObservationRuntimePolicyMismatch);
    }
    if observation.runtime_verifier_ref() != parent.runtime_verifier_ref() {
        issues.push(FreshMappedObservationIssue::ObservationVerifierMismatch);
    }
    if observation.backend_id() != parent.backend_id() {
        issues.push(FreshMappedObservationIssue::ObservationBackendMismatch);
    }
    if observation.host_identity_digest() != parent.host_identity_digest() {
        issues.push(FreshMappedObservationIssue::ObservationHostIdentityMismatch);
    }
    if observation.host_executable_path() != parent.executable_path() {
        issues.push(FreshMappedObservationIssue::ObservationExecutablePathMismatch);
    }
    if observation.host_executable_digest() != parent.executable_digest() {
        issues.push(FreshMappedObservationIssue::ObservationExecutableDigestMismatch);
    }
    if observation.closure_digest() != parent.dependency_closure_digest() {
        issues.push(FreshMappedObservationIssue::ObservationClosureDigestMismatch);
    }
    issues
}

fn fresh_base_report(
    policy: &FreshMappedRuntimePolicy,
    policy_digest: Option<String>,
    challenge: &MappedRuntimeCheckpointChallenge,
    challenge_digest: Option<String>,
    parent: &ClosureBackedInProcessContinuousVerifierExecution,
    mapped_runtime_policy_digest: &str,
    observed_at_ms: u64,
) -> FreshMappedObservationReport {
    FreshMappedObservationReport {
        schema_version: FRESH_MAPPED_RUNTIME_OBSERVATION_REPORT_SCHEMA_V1.into(),
        policy_id: policy.policy_id.clone(),
        policy_digest,
        challenge_digest,
        challenge_nonce_blake3_hex: challenge.nonce_blake3_hex.clone(),
        process_instance_id: challenge.process_instance_id.clone(),
        checkpoint_sequence: challenge.checkpoint_sequence,
        checkpoint_monotonic_counter: challenge.checkpoint_monotonic_counter,
        previous_checkpoint_digest: challenge.previous_checkpoint_digest.clone(),
        closure_backed_qualification_digest: parent.qualification_digest().into(),
        nix_binding_qualification_digest: parent.nix_binding_qualification_digest().into(),
        mapped_runtime_policy_digest: mapped_runtime_policy_digest.into(),
        raw_observation_qualification_digest: None,
        mapped_object_set_digest: None,
        runtime_policy_digest: parent.runtime_policy_digest().into(),
        runtime_verifier_ref: parent.runtime_verifier_ref().into(),
        backend_id: parent.backend_id().into(),
        host_identity_digest: parent.host_identity_digest().into(),
        executable_path: parent.executable_path().into(),
        executable_digest: parent.executable_digest().into(),
        dependency_closure_digest: parent.dependency_closure_digest().into(),
        observed_at_ms,
        disposition: FreshMappedObservationDisposition::Invalid,
        issues: Vec::new(),
    }
}

fn finalize_fresh(mut report: FreshMappedObservationReport) -> FreshMappedObservationReport {
    report.disposition = if report.issues.iter().any(FreshMappedObservationIssue::is_invalid) {
        FreshMappedObservationDisposition::Invalid
    } else {
        FreshMappedObservationDisposition::Blocked
    };
    report
}

fn fresh_observation_qualification_digest(
    policy_digest: &str,
    challenge_digest: &str,
    closure_backed_qualification_digest: &str,
    raw_observation_qualification_digest: &str,
    mapped_object_set_digest: &str,
    report_digest: &str,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(OBSERVATION_QUALIFICATION_DIGEST_DOMAIN);
    for field in [
        policy_digest,
        challenge_digest,
        closure_backed_qualification_digest,
        raw_observation_qualification_digest,
        mapped_object_set_digest,
        report_digest,
    ] {
        push_field_le(&mut hasher, field);
    }
    format!("blake3:{}", hasher.finalize().to_hex())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FreshMappedContinuityDisposition {
    Invalid,
    Blocked,
    Qualified,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FreshMappedContinuityIssue {
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
    DuplicateChallengeNonce { sequence: u64 },
    DuplicateFreshQualification { sequence: u64 },
    FreshPolicyMismatch { sequence: u64 },
    FreshParentMismatch { sequence: u64 },
    FreshNixBindingMismatch { sequence: u64 },
    FreshRuntimePolicyMismatch { sequence: u64 },
    FreshVerifierMismatch { sequence: u64 },
    FreshBackendMismatch { sequence: u64 },
    FreshProcessMismatch { sequence: u64 },
    ChallengeSequenceMismatch { sequence: u64 },
    ChallengeCounterMismatch { sequence: u64 },
    ChallengePredecessorMismatch { sequence: u64 },
    ObservationTimeMismatch { sequence: u64 },
    CheckpointVerifierMismatch { sequence: u64 },
    CheckpointProcessMismatch { sequence: u64 },
    CheckpointExecutableDigestMismatch { sequence: u64 },
    CheckpointClosureDigestMismatch { sequence: u64 },
    DynamicMeasurementDigestMismatch { sequence: u64 },
}

impl FreshMappedContinuityIssue {
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
            Self::DuplicateChallengeNonce { sequence } => {
                format!("duplicate-challenge-nonce:{sequence}")
            }
            Self::DuplicateFreshQualification { sequence } => {
                format!("duplicate-fresh-qualification:{sequence}")
            }
            Self::FreshPolicyMismatch { sequence } => format!("fresh-policy-mismatch:{sequence}"),
            Self::FreshParentMismatch { sequence } => format!("fresh-parent-mismatch:{sequence}"),
            Self::FreshNixBindingMismatch { sequence } => {
                format!("fresh-nix-binding-mismatch:{sequence}")
            }
            Self::FreshRuntimePolicyMismatch { sequence } => {
                format!("fresh-runtime-policy-mismatch:{sequence}")
            }
            Self::FreshVerifierMismatch { sequence } => {
                format!("fresh-verifier-mismatch:{sequence}")
            }
            Self::FreshBackendMismatch { sequence } => {
                format!("fresh-backend-mismatch:{sequence}")
            }
            Self::FreshProcessMismatch { sequence } => {
                format!("fresh-process-mismatch:{sequence}")
            }
            Self::ChallengeSequenceMismatch { sequence } => {
                format!("challenge-sequence-mismatch:{sequence}")
            }
            Self::ChallengeCounterMismatch { sequence } => {
                format!("challenge-counter-mismatch:{sequence}")
            }
            Self::ChallengePredecessorMismatch { sequence } => {
                format!("challenge-predecessor-mismatch:{sequence}")
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
pub struct FreshMappedCheckpointBinding {
    pub sequence: u64,
    pub monotonic_counter: u64,
    pub previous_checkpoint_digest: Option<String>,
    pub checkpoint_digest: String,
    pub checkpoint_dynamic_measurement_digest: String,
    pub challenge_digest: String,
    pub challenge_nonce_blake3_hex: String,
    pub fresh_observation_qualification_digest: String,
    pub raw_observation_qualification_digest: String,
    pub mapped_object_set_digest: String,
    pub checkpoint_observed_at_ms: u64,
    pub observation_observed_at_ms: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct FreshMappedContinuityReport {
    pub schema_version: String,
    pub policy_id: String,
    pub policy_digest: Option<String>,
    pub closure_backed_qualification_digest: String,
    pub continuous_execution_digest: String,
    pub runtime_policy_digest: String,
    pub runtime_verifier_ref: String,
    pub backend_id: String,
    pub process_instance_id: String,
    pub runtime_trace_digest: String,
    pub fresh_sequence_digest: Option<String>,
    pub checkpoint_count: u64,
    pub first_sequence: u64,
    pub last_sequence: u64,
    pub disposition: FreshMappedContinuityDisposition,
    pub issues: Vec<FreshMappedContinuityIssue>,
}

impl FreshMappedContinuityReport {
    pub fn canonical_digest(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(CONTINUITY_REPORT_DIGEST_DOMAIN);
        for field in [
            self.schema_version.as_str(),
            self.policy_id.as_str(),
            self.policy_digest.as_deref().unwrap_or("-"),
            self.closure_backed_qualification_digest.as_str(),
            self.continuous_execution_digest.as_str(),
            self.runtime_policy_digest.as_str(),
            self.runtime_verifier_ref.as_str(),
            self.backend_id.as_str(),
            self.process_instance_id.as_str(),
            self.runtime_trace_digest.as_str(),
            self.fresh_sequence_digest.as_deref().unwrap_or("-"),
        ] {
            push_field_le(&mut hasher, field);
        }
        hasher.update(&self.checkpoint_count.to_le_bytes());
        hasher.update(&self.first_sequence.to_le_bytes());
        hasher.update(&self.last_sequence.to_le_bytes());
        push_field_le(
            &mut hasher,
            match self.disposition {
                FreshMappedContinuityDisposition::Invalid => "invalid",
                FreshMappedContinuityDisposition::Blocked => "blocked",
                FreshMappedContinuityDisposition::Qualified => "qualified",
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
pub struct FreshMappedRuntimeContinuity {
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
    fresh_sequence_digest: String,
    bindings: Vec<FreshMappedCheckpointBinding>,
}

impl FreshMappedRuntimeContinuity {
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
    pub fn fresh_sequence_digest(&self) -> &str { &self.fresh_sequence_digest }
    pub fn bindings(&self) -> &[FreshMappedCheckpointBinding] { &self.bindings }
    pub const fn every_signed_checkpoint_commits_challenge_bound_live_observation(&self) -> bool {
        true
    }
    pub const fn checkpoint_lineage_coordinates_bound_into_every_observation(&self) -> bool {
        true
    }
    pub const fn challenge_nonces_unique_within_trace(&self) -> bool { true }
    pub const fn fresh_observation_qualifications_unique_within_trace(&self) -> bool { true }
    pub const fn intra_trace_cross_checkpoint_observation_reuse_excluded(&self) -> bool { true }
    pub const fn exact_signed_checkpoint_trace_rebound_to_parent_continuity(&self) -> bool { true }
    pub const fn trusted_challenge_authority_established(&self) -> bool { false }
    pub const fn challenge_nonce_unpredictability_established(&self) -> bool { false }
    pub const fn global_cross_trace_replay_excluded(&self) -> bool { false }
    pub const fn mapped_runtime_continuity_between_checkpoints_established(&self) -> bool { false }
    pub const fn mapping_continuity_since_exec_established(&self) -> bool { false }
    pub const fn closure_reverified_at_each_checkpoint(&self) -> bool { false }
    pub const fn dependency_closure_atomically_pinned(&self) -> bool { false }
    pub const fn root_resistant_immutability_established(&self) -> bool { false }
    pub const fn trusted_time_established(&self) -> bool { false }
    pub const fn grants_physical_authority(&self) -> bool { false }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FreshMappedRuntimeContinuityQualification {
    pub report: FreshMappedContinuityReport,
    verified: FreshMappedRuntimeContinuity,
}

impl FreshMappedRuntimeContinuityQualification {
    pub fn verified(&self) -> &FreshMappedRuntimeContinuity { &self.verified }
    pub fn into_verified(self) -> FreshMappedRuntimeContinuity { self.verified }
}

pub fn bind_fresh_mapped_runtime_continuity(
    policy: &FreshMappedRuntimePolicy,
    parent: &ClosureBackedInProcessContinuousVerifierExecution,
    continuous: &ContinuousVerifierExecution,
    evidence: &VerifierRuntimeEvidence,
    observations: &[FreshMappedRuntimeObservation],
) -> Result<FreshMappedRuntimeContinuityQualification, FreshMappedContinuityReport> {
    let policy_digest = policy.canonical_digest();
    let mut report = continuity_base_report(policy, policy_digest.clone(), parent);

    if !policy.validate() {
        report.issues.push(FreshMappedContinuityIssue::InvalidPolicy);
        return Err(finalize_continuity(report));
    }
    if !evidence.validate() || evidence.checkpoints.len() > MAX_CHECKPOINT_BINDINGS {
        report.issues.push(FreshMappedContinuityIssue::InvalidRuntimeEvidence);
        return Err(finalize_continuity(report));
    }
    if parent.policy_digest() != policy.expected_closure_backed_policy_digest {
        report.issues.push(FreshMappedContinuityIssue::ClosureBackedPolicyMismatch);
    }
    if parent.continuous_execution_digest() != continuous.continuity_digest() {
        report.issues.push(FreshMappedContinuityIssue::ContinuousExecutionMismatch);
    }
    if parent.runtime_policy_digest() != policy.expected_runtime_policy_digest
        || continuous.policy_digest() != policy.expected_runtime_policy_digest
    {
        report.issues.push(FreshMappedContinuityIssue::RuntimePolicyMismatch);
    }
    if parent.runtime_verifier_ref() != policy.expected_runtime_verifier_ref
        || continuous.verifier_ref() != policy.expected_runtime_verifier_ref
    {
        report.issues.push(FreshMappedContinuityIssue::RuntimeVerifierMismatch);
    }
    if parent.backend_id() != policy.expected_backend_id {
        report.issues.push(FreshMappedContinuityIssue::BackendMismatch);
    }
    if parent.process_instance_id() != continuous.process_instance_id() {
        report.issues.push(FreshMappedContinuityIssue::ProcessInstanceMismatch);
    }

    let rebound = match rebind_runtime_evidence(evidence, continuous.assessed_at_ms()) {
        Some(value) => value,
        None => {
            report.issues.push(FreshMappedContinuityIssue::InvalidRuntimeEvidence);
            return Err(finalize_continuity(report));
        }
    };
    if rebound.runtime_trace_digest != continuous.runtime_trace_digest()
        || rebound.runtime_trace_digest != parent.runtime_trace_digest()
    {
        report.issues.push(FreshMappedContinuityIssue::RuntimeTraceMismatch);
    }
    if rebound.launch_digest != continuous.launch_attestation_digest() {
        report.issues.push(FreshMappedContinuityIssue::LaunchDigestMismatch);
    }
    if rebound.final_checkpoint_digest != continuous.final_checkpoint_digest() {
        report.issues.push(FreshMappedContinuityIssue::FinalCheckpointDigestMismatch);
    }
    if rebound.computation_digest != continuous.computation_attestation_digest() {
        report.issues.push(FreshMappedContinuityIssue::ComputationDigestMismatch);
    }
    if evidence.computation.input_digest != continuous.input_digest() {
        report.issues.push(FreshMappedContinuityIssue::InputDigestMismatch);
    }
    if evidence.computation.output_digest != continuous.output_digest() {
        report.issues.push(FreshMappedContinuityIssue::OutputDigestMismatch);
    }
    if evidence.computation.started_at_ms != continuous.started_at_ms()
        || evidence.computation.completed_at_ms != continuous.completed_at_ms()
    {
        report.issues.push(FreshMappedContinuityIssue::ComputationTimeMismatch);
    }
    if continuous.assessed_at_ms() != parent.continuous_assessed_at_ms() {
        report.issues.push(FreshMappedContinuityIssue::AssessmentTimeMismatch);
    }
    if evidence.checkpoints.len() != observations.len() {
        report.issues.push(FreshMappedContinuityIssue::ObservationCountMismatch {
            checkpoints: evidence.checkpoints.len() as u64,
            observations: observations.len() as u64,
        });
    }
    if !report.issues.is_empty() {
        return Err(finalize_continuity(report));
    }

    let mut nonces = BTreeSet::new();
    let mut qualifications = BTreeSet::new();
    let mut bindings = Vec::with_capacity(evidence.checkpoints.len());

    for (index, (checkpoint, observation)) in evidence
        .checkpoints
        .iter()
        .zip(observations.iter())
        .enumerate()
    {
        let sequence = index as u64 + 1;
        if !nonces.insert(observation.challenge_nonce_blake3_hex()) {
            report
                .issues
                .push(FreshMappedContinuityIssue::DuplicateChallengeNonce { sequence });
        }
        if !qualifications.insert(observation.qualification_digest()) {
            report
                .issues
                .push(FreshMappedContinuityIssue::DuplicateFreshQualification { sequence });
        }
        report.issues.extend(fresh_checkpoint_issues(
            policy,
            parent,
            continuous,
            checkpoint.verifier_ref.as_str(),
            checkpoint.process_instance_id.as_str(),
            checkpoint.sequence,
            checkpoint.monotonic_counter,
            checkpoint.previous_checkpoint_digest.as_deref(),
            checkpoint.executable_digest.as_str(),
            checkpoint.dependency_closure_digest.as_str(),
            checkpoint.dynamic_measurement_digest.as_str(),
            checkpoint.observed_at_ms,
            observation,
            sequence,
        ));

        bindings.push(FreshMappedCheckpointBinding {
            sequence: checkpoint.sequence,
            monotonic_counter: checkpoint.monotonic_counter,
            previous_checkpoint_digest: checkpoint.previous_checkpoint_digest.clone(),
            checkpoint_digest: rebound.checkpoint_digests[index].clone(),
            checkpoint_dynamic_measurement_digest: checkpoint.dynamic_measurement_digest.clone(),
            challenge_digest: observation.challenge_digest().into(),
            challenge_nonce_blake3_hex: observation.challenge_nonce_blake3_hex().into(),
            fresh_observation_qualification_digest: observation.qualification_digest().into(),
            raw_observation_qualification_digest: observation
                .raw_observation_qualification_digest()
                .into(),
            mapped_object_set_digest: observation.mapped_object_set_digest().into(),
            checkpoint_observed_at_ms: checkpoint.observed_at_ms,
            observation_observed_at_ms: observation.observed_at_ms(),
        });
    }

    if !report.issues.is_empty() {
        return Err(finalize_continuity(report));
    }

    let sequence_digest = fresh_sequence_digest(&bindings);
    report.fresh_sequence_digest = Some(sequence_digest.clone());
    report.checkpoint_count = bindings.len() as u64;
    report.first_sequence = bindings.first().map(|value| value.sequence).unwrap_or(0);
    report.last_sequence = bindings.last().map(|value| value.sequence).unwrap_or(0);
    report.disposition = FreshMappedContinuityDisposition::Qualified;

    let report_digest = report.canonical_digest();
    let policy_digest = policy_digest.expect("validated policy has digest");
    let qualification_digest = continuity_qualification_digest(
        &policy_digest,
        parent.qualification_digest(),
        continuous.continuity_digest(),
        &sequence_digest,
        &report_digest,
    );
    let verified = FreshMappedRuntimeContinuity {
        qualification_digest,
        report_digest,
        policy_digest,
        closure_backed_qualification_digest: parent.qualification_digest().into(),
        continuous_execution_digest: continuous.continuity_digest().into(),
        runtime_policy_digest: parent.runtime_policy_digest().into(),
        runtime_verifier_ref: parent.runtime_verifier_ref().into(),
        backend_id: parent.backend_id().into(),
        process_instance_id: parent.process_instance_id().into(),
        runtime_trace_digest: parent.runtime_trace_digest().into(),
        fresh_sequence_digest: sequence_digest,
        bindings,
    };

    Ok(FreshMappedRuntimeContinuityQualification { report, verified })
}

#[allow(clippy::too_many_arguments)]
fn fresh_checkpoint_issues(
    policy: &FreshMappedRuntimePolicy,
    parent: &ClosureBackedInProcessContinuousVerifierExecution,
    continuous: &ContinuousVerifierExecution,
    checkpoint_verifier_ref: &str,
    checkpoint_process_instance_id: &str,
    checkpoint_sequence: u64,
    checkpoint_monotonic_counter: u64,
    checkpoint_previous_digest: Option<&str>,
    checkpoint_executable_digest: &str,
    checkpoint_closure_digest: &str,
    checkpoint_dynamic_measurement_digest: &str,
    checkpoint_observed_at_ms: u64,
    observation: &FreshMappedRuntimeObservation,
    sequence: u64,
) -> Vec<FreshMappedContinuityIssue> {
    let mut issues = Vec::new();
    if observation.policy_digest() != policy.canonical_digest().as_deref().unwrap_or("-") {
        issues.push(FreshMappedContinuityIssue::FreshPolicyMismatch { sequence });
    }
    if observation.closure_backed_qualification_digest() != parent.qualification_digest() {
        issues.push(FreshMappedContinuityIssue::FreshParentMismatch { sequence });
    }
    if observation.nix_binding_qualification_digest() != parent.nix_binding_qualification_digest() {
        issues.push(FreshMappedContinuityIssue::FreshNixBindingMismatch { sequence });
    }
    if observation.runtime_policy_digest() != parent.runtime_policy_digest() {
        issues.push(FreshMappedContinuityIssue::FreshRuntimePolicyMismatch { sequence });
    }
    if observation.runtime_verifier_ref() != parent.runtime_verifier_ref() {
        issues.push(FreshMappedContinuityIssue::FreshVerifierMismatch { sequence });
    }
    if observation.backend_id() != parent.backend_id() {
        issues.push(FreshMappedContinuityIssue::FreshBackendMismatch { sequence });
    }
    if observation.process_instance_id() != continuous.process_instance_id() {
        issues.push(FreshMappedContinuityIssue::FreshProcessMismatch { sequence });
    }
    if observation.checkpoint_sequence() != checkpoint_sequence || checkpoint_sequence != sequence {
        issues.push(FreshMappedContinuityIssue::ChallengeSequenceMismatch { sequence });
    }
    if observation.checkpoint_monotonic_counter() != checkpoint_monotonic_counter {
        issues.push(FreshMappedContinuityIssue::ChallengeCounterMismatch { sequence });
    }
    if observation.previous_checkpoint_digest() != checkpoint_previous_digest {
        issues.push(FreshMappedContinuityIssue::ChallengePredecessorMismatch { sequence });
    }
    if observation.observed_at_ms() != checkpoint_observed_at_ms {
        issues.push(FreshMappedContinuityIssue::ObservationTimeMismatch { sequence });
    }
    if checkpoint_verifier_ref != continuous.verifier_ref() {
        issues.push(FreshMappedContinuityIssue::CheckpointVerifierMismatch { sequence });
    }
    if checkpoint_process_instance_id != continuous.process_instance_id() {
        issues.push(FreshMappedContinuityIssue::CheckpointProcessMismatch { sequence });
    }
    if checkpoint_executable_digest != parent.executable_digest() {
        issues.push(FreshMappedContinuityIssue::CheckpointExecutableDigestMismatch { sequence });
    }
    if checkpoint_closure_digest != parent.dependency_closure_digest() {
        issues.push(FreshMappedContinuityIssue::CheckpointClosureDigestMismatch { sequence });
    }
    if checkpoint_dynamic_measurement_digest != observation.qualification_digest() {
        issues.push(FreshMappedContinuityIssue::DynamicMeasurementDigestMismatch { sequence });
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

fn fresh_sequence_digest(bindings: &[FreshMappedCheckpointBinding]) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(FRESH_SEQUENCE_DIGEST_DOMAIN);
    hasher.update(&(bindings.len() as u64).to_le_bytes());
    for binding in bindings {
        hasher.update(&binding.sequence.to_le_bytes());
        hasher.update(&binding.monotonic_counter.to_le_bytes());
        push_optional_field_le(&mut hasher, binding.previous_checkpoint_digest.as_deref());
        for field in [
            binding.checkpoint_digest.as_str(),
            binding.checkpoint_dynamic_measurement_digest.as_str(),
            binding.challenge_digest.as_str(),
            binding.challenge_nonce_blake3_hex.as_str(),
            binding.fresh_observation_qualification_digest.as_str(),
            binding.raw_observation_qualification_digest.as_str(),
            binding.mapped_object_set_digest.as_str(),
        ] {
            push_field_le(&mut hasher, field);
        }
        hasher.update(&binding.checkpoint_observed_at_ms.to_le_bytes());
        hasher.update(&binding.observation_observed_at_ms.to_le_bytes());
    }
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn continuity_qualification_digest(
    policy_digest: &str,
    closure_backed_qualification_digest: &str,
    continuous_execution_digest: &str,
    fresh_sequence_digest: &str,
    report_digest: &str,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(CONTINUITY_QUALIFICATION_DIGEST_DOMAIN);
    for field in [
        policy_digest,
        closure_backed_qualification_digest,
        continuous_execution_digest,
        fresh_sequence_digest,
        report_digest,
    ] {
        push_field_le(&mut hasher, field);
    }
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn continuity_base_report(
    policy: &FreshMappedRuntimePolicy,
    policy_digest: Option<String>,
    parent: &ClosureBackedInProcessContinuousVerifierExecution,
) -> FreshMappedContinuityReport {
    FreshMappedContinuityReport {
        schema_version: FRESH_MAPPED_RUNTIME_CONTINUITY_REPORT_SCHEMA_V1.into(),
        policy_id: policy.policy_id.clone(),
        policy_digest,
        closure_backed_qualification_digest: parent.qualification_digest().into(),
        continuous_execution_digest: parent.continuous_execution_digest().into(),
        runtime_policy_digest: parent.runtime_policy_digest().into(),
        runtime_verifier_ref: parent.runtime_verifier_ref().into(),
        backend_id: parent.backend_id().into(),
        process_instance_id: parent.process_instance_id().into(),
        runtime_trace_digest: parent.runtime_trace_digest().into(),
        fresh_sequence_digest: None,
        checkpoint_count: 0,
        first_sequence: 0,
        last_sequence: 0,
        disposition: FreshMappedContinuityDisposition::Invalid,
        issues: Vec::new(),
    }
}

fn finalize_continuity(mut report: FreshMappedContinuityReport) -> FreshMappedContinuityReport {
    report.disposition = if report.issues.iter().any(FreshMappedContinuityIssue::is_invalid) {
        FreshMappedContinuityDisposition::Invalid
    } else {
        FreshMappedContinuityDisposition::Blocked
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
    lower_hex_exact(hex, 64)
}

fn lower_hex_exact(value: &str, len: usize) -> bool {
    value.len() == len
        && value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn valid_refs(refs: &[String]) -> bool {
    refs.len() <= MAX_EVIDENCE_REFS
        && refs.iter().all(|value| canonical_text(value))
        && unique_strings(refs)
}

fn unique_strings(values: &[String]) -> bool {
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

fn push_optional_field_le(hasher: &mut blake3::Hasher, value: Option<&str>) {
    match value {
        Some(value) => {
            hasher.update(&[1]);
            push_field_le(hasher, value);
        }
        None => hasher.update(&[0]),
    }
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

    fn policy() -> FreshMappedRuntimePolicy {
        FreshMappedRuntimePolicy {
            schema_version: FRESH_MAPPED_RUNTIME_POLICY_SCHEMA_V1.into(),
            policy_id: "policy:fresh-mapped-runtime:1".into(),
            expected_closure_backed_policy_digest: d("closure-backed-policy"),
            expected_nix_binding_policy_digest: d("nix-binding-policy"),
            expected_mapped_runtime_policy_digest: d("mapped-policy"),
            expected_runtime_policy_digest: d("runtime-policy"),
            expected_runtime_verifier_ref: "verifier:host".into(),
            expected_backend_id: "backend:in-process".into(),
            evidence_refs: vec!["review:a".into(), "review:b".into()],
        }
    }

    fn challenge(sequence: u64, counter: u64, predecessor: Option<String>, nonce: u8) -> MappedRuntimeCheckpointChallenge {
        MappedRuntimeCheckpointChallenge {
            schema_version: MAPPED_RUNTIME_CHECKPOINT_CHALLENGE_SCHEMA_V1.into(),
            process_instance_id: "process:1".into(),
            runtime_policy_digest: d("runtime-policy"),
            verifier_ref: "verifier:host".into(),
            checkpoint_sequence: sequence,
            checkpoint_monotonic_counter: counter,
            previous_checkpoint_digest: predecessor,
            nonce_blake3_hex: format!("{nonce:02x}").repeat(32),
        }
    }

    fn binding(sequence: u64, nonce: u8) -> FreshMappedCheckpointBinding {
        FreshMappedCheckpointBinding {
            sequence,
            monotonic_counter: 100 + sequence,
            previous_checkpoint_digest: if sequence == 1 {
                None
            } else {
                Some(d(&format!("checkpoint:{}", sequence - 1)))
            },
            checkpoint_digest: d(&format!("checkpoint:{sequence}")),
            checkpoint_dynamic_measurement_digest: d(&format!("fresh:{sequence}")),
            challenge_digest: d(&format!("challenge:{sequence}")),
            challenge_nonce_blake3_hex: format!("{nonce:02x}").repeat(32),
            fresh_observation_qualification_digest: d(&format!("fresh:{sequence}")),
            raw_observation_qualification_digest: d(&format!("raw:{sequence}")),
            mapped_object_set_digest: d(&format!("mapped:{sequence}")),
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
    fn challenge_binds_lineage_coordinates_and_nonce() {
        let first = challenge(1, 10, None, 1);
        assert!(first.validate());
        let mut changed = first.clone();
        changed.checkpoint_monotonic_counter += 1;
        assert_ne!(first.canonical_digest(), changed.canonical_digest());
        let mut changed_nonce = first.clone();
        changed_nonce.nonce_blake3_hex = "02".repeat(32);
        assert_ne!(first.canonical_digest(), changed_nonce.canonical_digest());
    }

    #[test]
    fn challenge_requires_predecessor_after_first_checkpoint() {
        assert!(!challenge(2, 11, None, 3).validate());
        assert!(challenge(2, 11, Some(d("checkpoint:1")), 3).validate());
        assert!(!challenge(1, 10, Some(d("checkpoint:0")), 3).validate());
    }

    #[test]
    fn fresh_sequence_digest_binds_nonce_counter_predecessor_and_order() {
        let left = vec![binding(1, 1), binding(2, 2)];
        let reversed = vec![binding(2, 2), binding(1, 1)];
        assert_ne!(fresh_sequence_digest(&left), fresh_sequence_digest(&reversed));

        let mut changed = left.clone();
        changed[1].challenge_nonce_blake3_hex = "03".repeat(32);
        assert_ne!(fresh_sequence_digest(&left), fresh_sequence_digest(&changed));

        let mut changed_counter = left.clone();
        changed_counter[1].monotonic_counter += 1;
        assert_ne!(fresh_sequence_digest(&left), fresh_sequence_digest(&changed_counter));
    }

    #[test]
    fn duplicate_nonce_and_qualification_are_detectable() {
        let values = [binding(1, 7), binding(2, 7)];
        let mut nonces = BTreeSet::new();
        assert!(nonces.insert(values[0].challenge_nonce_blake3_hex.as_str()));
        assert!(!nonces.insert(values[1].challenge_nonce_blake3_hex.as_str()));

        let mut qualifications = BTreeSet::new();
        assert!(qualifications.insert(values[0].fresh_observation_qualification_digest.as_str()));
        let same = values[0].fresh_observation_qualification_digest.clone();
        assert!(!qualifications.insert(same.as_str()));
    }

    #[test]
    fn runtime_trace_digest_matches_parent_big_endian_encoding() {
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
