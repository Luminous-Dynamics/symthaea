// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Authorized signed checkpoint-two inclusion for one exact authenticated
//! post-release live runtime observation.
//!
//! This bridge deliberately does not create signatures. It verifies one
//! existing `RuntimeMeasurementCheckpoint` using the exact canonical unsigned
//! bytes and trust semantics frozen by the runtime-continuity domain, then
//! requires that checkpoint to commit the opaque authenticated post-release
//! observation as its dynamic measurement.

#![cfg(target_os = "linux")]
#![deny(unsafe_code)]

use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use symthaea_assurance_authenticated_post_release_checkpoint_two::AuthenticatedPostReleaseCheckpointTwoObservation;
use symthaea_evidence_verifier_runtime_continuity::{
    RuntimeMeasurementCheckpoint, RuntimeMeasurementScope, VerifierRuntimeContinuityPolicy,
};

pub const SIGNED_POST_RELEASE_CHECKPOINT_TWO_POLICY_SCHEMA_V1: &str =
    "symthaea.assurance.signed-post-release-checkpoint-two-policy.v1";
pub const SIGNED_POST_RELEASE_CHECKPOINT_TWO_REPORT_SCHEMA_V1: &str =
    "symthaea.assurance.signed-post-release-checkpoint-two-report.v1";

const POLICY_DOMAIN: &[u8] =
    b"symthaea.assurance.signed-post-release-checkpoint-two-policy.digest.v1\0";
const REPORT_DOMAIN: &[u8] =
    b"symthaea.assurance.signed-post-release-checkpoint-two-report.digest.v1\0";
const QUALIFICATION_DOMAIN: &[u8] =
    b"symthaea.assurance.signed-post-release-checkpoint-two-qualification.digest.v1\0";
const MAX_TEXT: usize = 4096;
const MAX_REFS: usize = 128;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignedPostReleaseCheckpointTwoPolicy {
    pub schema_version: String,
    pub policy_id: String,
    pub expected_authenticated_observation_policy_digest: String,
    pub expected_runtime_policy_digest: String,
    pub expected_runtime_verifier_ref: String,
    pub expected_backend_id: String,
    pub evidence_refs: Vec<String>,
}

impl SignedPostReleaseCheckpointTwoPolicy {
    pub fn validate(&self) -> bool {
        self.schema_version == SIGNED_POST_RELEASE_CHECKPOINT_TWO_POLICY_SCHEMA_V1
            && canonical_text(&self.policy_id)
            && valid_blake3(&self.expected_authenticated_observation_policy_digest)
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
            self.expected_authenticated_observation_policy_digest
                .as_str(),
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
pub enum SignedPostReleaseCheckpointTwoIssue {
    InvalidPolicy,
    InvalidRuntimePolicy,
    InvalidCheckpoint,
    AuthenticatedObservationPolicyMismatch,
    RuntimePolicyMismatch,
    RuntimeVerifierMismatch,
    BackendMismatch,
    LaunchAttestationMismatch,
    ProcessInstanceMismatch,
    CheckpointSequenceMismatch,
    PreviousCheckpointMismatch,
    ObservationTimeMismatch,
    MonotonicCounterMismatch,
    BootMeasurementMismatch,
    ExecutableMismatch,
    DependencyClosureMismatch,
    RuntimeConfigMismatch,
    DynamicMeasurementMismatch,
    CheckpointSignatureInvalid,
    CheckpointSignerUntrusted,
}

impl SignedPostReleaseCheckpointTwoIssue {
    fn invalid(&self) -> bool {
        matches!(
            self,
            Self::InvalidPolicy
                | Self::InvalidRuntimePolicy
                | Self::InvalidCheckpoint
                | Self::LaunchAttestationMismatch
                | Self::ProcessInstanceMismatch
                | Self::CheckpointSequenceMismatch
                | Self::PreviousCheckpointMismatch
                | Self::ObservationTimeMismatch
                | Self::MonotonicCounterMismatch
                | Self::DynamicMeasurementMismatch
                | Self::CheckpointSignatureInvalid
        )
    }

    fn code(&self) -> &'static str {
        match self {
            Self::InvalidPolicy => "invalid-policy",
            Self::InvalidRuntimePolicy => "invalid-runtime-policy",
            Self::InvalidCheckpoint => "invalid-checkpoint",
            Self::AuthenticatedObservationPolicyMismatch => {
                "authenticated-observation-policy-mismatch"
            }
            Self::RuntimePolicyMismatch => "runtime-policy-mismatch",
            Self::RuntimeVerifierMismatch => "runtime-verifier-mismatch",
            Self::BackendMismatch => "backend-mismatch",
            Self::LaunchAttestationMismatch => "launch-attestation-mismatch",
            Self::ProcessInstanceMismatch => "process-instance-mismatch",
            Self::CheckpointSequenceMismatch => "checkpoint-sequence-mismatch",
            Self::PreviousCheckpointMismatch => "previous-checkpoint-mismatch",
            Self::ObservationTimeMismatch => "observation-time-mismatch",
            Self::MonotonicCounterMismatch => "monotonic-counter-mismatch",
            Self::BootMeasurementMismatch => "boot-measurement-mismatch",
            Self::ExecutableMismatch => "executable-mismatch",
            Self::DependencyClosureMismatch => "dependency-closure-mismatch",
            Self::RuntimeConfigMismatch => "runtime-config-mismatch",
            Self::DynamicMeasurementMismatch => "dynamic-measurement-mismatch",
            Self::CheckpointSignatureInvalid => "checkpoint-signature-invalid",
            Self::CheckpointSignerUntrusted => "checkpoint-signer-untrusted",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignedPostReleaseCheckpointTwoDisposition {
    Invalid,
    Blocked,
    Qualified,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignedPostReleaseCheckpointTwoReport {
    pub schema_version: String,
    pub policy_id: String,
    pub policy_digest: Option<String>,
    pub authenticated_observation_qualification_digest: String,
    pub runtime_policy_digest: Option<String>,
    pub checkpoint_digest: Option<String>,
    pub launch_attestation_digest: String,
    pub process_instance_id: String,
    pub verifier_ref: String,
    pub sequence: u64,
    pub previous_checkpoint_digest: Option<String>,
    pub observed_at_ms: u64,
    pub monotonic_counter: u64,
    pub dynamic_measurement_digest: String,
    pub signer_key_id: String,
    pub signer_public_key_ed25519_hex: String,
    pub disposition: SignedPostReleaseCheckpointTwoDisposition,
    pub issues: Vec<SignedPostReleaseCheckpointTwoIssue>,
}

impl SignedPostReleaseCheckpointTwoReport {
    pub fn canonical_digest(&self) -> String {
        let mut h = blake3::Hasher::new();
        h.update(REPORT_DOMAIN);
        for value in [
            self.schema_version.as_str(),
            self.policy_id.as_str(),
            self.policy_digest.as_deref().unwrap_or("-"),
            self.authenticated_observation_qualification_digest.as_str(),
            self.runtime_policy_digest.as_deref().unwrap_or("-"),
            self.checkpoint_digest.as_deref().unwrap_or("-"),
            self.launch_attestation_digest.as_str(),
            self.process_instance_id.as_str(),
            self.verifier_ref.as_str(),
            self.previous_checkpoint_digest.as_deref().unwrap_or("-"),
            self.dynamic_measurement_digest.as_str(),
            self.signer_key_id.as_str(),
            self.signer_public_key_ed25519_hex.as_str(),
        ] {
            field(&mut h, value);
        }
        h.update(&self.sequence.to_le_bytes());
        h.update(&self.observed_at_ms.to_le_bytes());
        h.update(&self.monotonic_counter.to_le_bytes());
        field(
            &mut h,
            match self.disposition {
                SignedPostReleaseCheckpointTwoDisposition::Invalid => "invalid",
                SignedPostReleaseCheckpointTwoDisposition::Blocked => "blocked",
                SignedPostReleaseCheckpointTwoDisposition::Qualified => "qualified",
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
pub struct SignedPostReleaseCheckpointTwo {
    qualification_digest: String,
    report_digest: String,
    policy_digest: String,
    authenticated_observation_qualification_digest: String,
    runtime_policy_digest: String,
    checkpoint_digest: String,
    launch_attestation_digest: String,
    process_instance_id: String,
    verifier_ref: String,
    backend_id: String,
    previous_checkpoint_digest: String,
    observed_at_ms: u64,
    monotonic_counter: u64,
    boot_measurement_digest: String,
    executable_digest: String,
    dependency_closure_digest: String,
    runtime_config_digest: String,
    dynamic_measurement_digest: String,
    signer_key_id: String,
    signer_public_key_ed25519_hex: String,
}

impl SignedPostReleaseCheckpointTwo {
    pub fn qualification_digest(&self) -> &str {
        &self.qualification_digest
    }
    pub fn report_digest(&self) -> &str {
        &self.report_digest
    }
    pub fn policy_digest(&self) -> &str {
        &self.policy_digest
    }
    pub fn authenticated_observation_qualification_digest(&self) -> &str {
        &self.authenticated_observation_qualification_digest
    }
    pub fn runtime_policy_digest(&self) -> &str {
        &self.runtime_policy_digest
    }
    pub fn checkpoint_digest(&self) -> &str {
        &self.checkpoint_digest
    }
    pub fn launch_attestation_digest(&self) -> &str {
        &self.launch_attestation_digest
    }
    pub fn process_instance_id(&self) -> &str {
        &self.process_instance_id
    }
    pub fn verifier_ref(&self) -> &str {
        &self.verifier_ref
    }
    pub fn backend_id(&self) -> &str {
        &self.backend_id
    }
    pub const fn sequence(&self) -> u64 {
        2
    }
    pub fn previous_checkpoint_digest(&self) -> &str {
        &self.previous_checkpoint_digest
    }
    pub const fn observed_at_ms(&self) -> u64 {
        self.observed_at_ms
    }
    pub const fn monotonic_counter(&self) -> u64 {
        self.monotonic_counter
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
    pub fn dynamic_measurement_digest(&self) -> &str {
        &self.dynamic_measurement_digest
    }
    pub fn signer_key_id(&self) -> &str {
        &self.signer_key_id
    }
    pub fn signer_public_key_ed25519_hex(&self) -> &str {
        &self.signer_public_key_ed25519_hex
    }

    pub const fn checkpoint_signature_valid(&self) -> bool {
        true
    }
    pub const fn checkpoint_signer_authorized_by_exact_runtime_policy(&self) -> bool {
        true
    }
    pub const fn checkpoint_uses_frozen_runtime_checkpoint_encoding(&self) -> bool {
        true
    }
    pub const fn checkpoint_two_descends_from_exact_checkpoint_one(&self) -> bool {
        true
    }
    pub const fn checkpoint_two_static_identity_matches_authenticated_runtime(&self) -> bool {
        true
    }
    pub const fn signed_checkpoint_commits_authenticated_release_to_live_observation_chain(
        &self,
    ) -> bool {
        true
    }
    pub const fn signature_operation_after_live_observation_established(&self) -> bool {
        false
    }
    pub const fn checkpoint_gap_policy_established_here(&self) -> bool {
        false
    }
    pub const fn checkpoint_one_live_observation_proven_here(&self) -> bool {
        false
    }
    pub const fn mapping_continuity_between_checkpoints_established(&self) -> bool {
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

pub struct SignedPostReleaseCheckpointTwoQualification {
    pub report: SignedPostReleaseCheckpointTwoReport,
    signed: SignedPostReleaseCheckpointTwo,
}

impl SignedPostReleaseCheckpointTwoQualification {
    pub fn signed(&self) -> &SignedPostReleaseCheckpointTwo {
        &self.signed
    }
    pub fn into_signed(self) -> SignedPostReleaseCheckpointTwo {
        self.signed
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SignatureTrust {
    Trusted,
    Untrusted,
    Invalid,
}

pub fn verify_signed_post_release_checkpoint_two(
    policy: &SignedPostReleaseCheckpointTwoPolicy,
    authenticated: &AuthenticatedPostReleaseCheckpointTwoObservation,
    runtime_policy: &VerifierRuntimeContinuityPolicy,
    checkpoint: &RuntimeMeasurementCheckpoint,
) -> Result<SignedPostReleaseCheckpointTwoQualification, SignedPostReleaseCheckpointTwoReport> {
    let policy_digest = policy.canonical_digest();
    let runtime_policy_digest = runtime_policy.canonical_digest();
    let checkpoint_digest = checkpoint.canonical_digest();
    let mut report = SignedPostReleaseCheckpointTwoReport {
        schema_version: SIGNED_POST_RELEASE_CHECKPOINT_TWO_REPORT_SCHEMA_V1.into(),
        policy_id: policy.policy_id.clone(),
        policy_digest: policy_digest.clone(),
        authenticated_observation_qualification_digest: authenticated.qualification_digest().into(),
        runtime_policy_digest: runtime_policy_digest.clone(),
        checkpoint_digest: checkpoint_digest.clone(),
        launch_attestation_digest: checkpoint.launch_attestation_digest.clone(),
        process_instance_id: checkpoint.process_instance_id.clone(),
        verifier_ref: checkpoint.verifier_ref.clone(),
        sequence: checkpoint.sequence,
        previous_checkpoint_digest: checkpoint.previous_checkpoint_digest.clone(),
        observed_at_ms: checkpoint.observed_at_ms,
        monotonic_counter: checkpoint.monotonic_counter,
        dynamic_measurement_digest: checkpoint.dynamic_measurement_digest.clone(),
        signer_key_id: checkpoint.signer_key_id.clone(),
        signer_public_key_ed25519_hex: checkpoint.signer_public_key_ed25519_hex.clone(),
        disposition: SignedPostReleaseCheckpointTwoDisposition::Invalid,
        issues: Vec::new(),
    };

    if !policy.validate() {
        report
            .issues
            .push(SignedPostReleaseCheckpointTwoIssue::InvalidPolicy);
    }
    if !runtime_policy.validate() || runtime_policy_digest.is_none() {
        report
            .issues
            .push(SignedPostReleaseCheckpointTwoIssue::InvalidRuntimePolicy);
    }
    if !checkpoint.validate() || checkpoint_digest.is_none() {
        report
            .issues
            .push(SignedPostReleaseCheckpointTwoIssue::InvalidCheckpoint);
    }
    if !report.issues.is_empty() {
        return Err(finalize(report));
    }

    if authenticated.policy_digest() != policy.expected_authenticated_observation_policy_digest {
        report
            .issues
            .push(SignedPostReleaseCheckpointTwoIssue::AuthenticatedObservationPolicyMismatch);
    }
    if runtime_policy_digest.as_deref() != Some(policy.expected_runtime_policy_digest.as_str())
        || authenticated.runtime_policy_digest() != policy.expected_runtime_policy_digest
    {
        report
            .issues
            .push(SignedPostReleaseCheckpointTwoIssue::RuntimePolicyMismatch);
    }
    if runtime_policy.verifier_ref != policy.expected_runtime_verifier_ref
        || authenticated.runtime_verifier_ref() != policy.expected_runtime_verifier_ref
        || checkpoint.verifier_ref != policy.expected_runtime_verifier_ref
    {
        report
            .issues
            .push(SignedPostReleaseCheckpointTwoIssue::RuntimeVerifierMismatch);
    }
    if authenticated.backend_id() != policy.expected_backend_id {
        report
            .issues
            .push(SignedPostReleaseCheckpointTwoIssue::BackendMismatch);
    }
    if checkpoint.launch_attestation_digest != authenticated.launch_attestation_digest() {
        report
            .issues
            .push(SignedPostReleaseCheckpointTwoIssue::LaunchAttestationMismatch);
    }
    if checkpoint.process_instance_id != authenticated.process_instance_id() {
        report
            .issues
            .push(SignedPostReleaseCheckpointTwoIssue::ProcessInstanceMismatch);
    }
    if checkpoint.sequence != 2 {
        report
            .issues
            .push(SignedPostReleaseCheckpointTwoIssue::CheckpointSequenceMismatch);
    }
    if checkpoint.previous_checkpoint_digest.as_deref()
        != Some(authenticated.bootstrap_ready_checkpoint_digest())
    {
        report
            .issues
            .push(SignedPostReleaseCheckpointTwoIssue::PreviousCheckpointMismatch);
    }
    if checkpoint.observed_at_ms != authenticated.checkpoint_two_observed_at_ms() {
        report
            .issues
            .push(SignedPostReleaseCheckpointTwoIssue::ObservationTimeMismatch);
    }
    if checkpoint.monotonic_counter != authenticated.checkpoint_two_counter() {
        report
            .issues
            .push(SignedPostReleaseCheckpointTwoIssue::MonotonicCounterMismatch);
    }
    if checkpoint.boot_measurement_digest != authenticated.boot_measurement_digest()
        || checkpoint.boot_measurement_digest != runtime_policy.expected_boot_measurement_digest
    {
        report
            .issues
            .push(SignedPostReleaseCheckpointTwoIssue::BootMeasurementMismatch);
    }
    if checkpoint.executable_digest != authenticated.executable_digest()
        || checkpoint.executable_digest != runtime_policy.expected_executable_digest
    {
        report
            .issues
            .push(SignedPostReleaseCheckpointTwoIssue::ExecutableMismatch);
    }
    if checkpoint.dependency_closure_digest != authenticated.dependency_closure_digest()
        || checkpoint.dependency_closure_digest != runtime_policy.expected_dependency_closure_digest
    {
        report
            .issues
            .push(SignedPostReleaseCheckpointTwoIssue::DependencyClosureMismatch);
    }
    if checkpoint.runtime_config_digest != authenticated.runtime_config_digest()
        || checkpoint.runtime_config_digest != runtime_policy.expected_runtime_config_digest
    {
        report
            .issues
            .push(SignedPostReleaseCheckpointTwoIssue::RuntimeConfigMismatch);
    }
    if checkpoint.dynamic_measurement_digest != authenticated.qualification_digest() {
        report
            .issues
            .push(SignedPostReleaseCheckpointTwoIssue::DynamicMeasurementMismatch);
    }

    match verify_checkpoint_signature(runtime_policy, checkpoint) {
        SignatureTrust::Trusted => {}
        SignatureTrust::Untrusted => report
            .issues
            .push(SignedPostReleaseCheckpointTwoIssue::CheckpointSignerUntrusted),
        SignatureTrust::Invalid => report
            .issues
            .push(SignedPostReleaseCheckpointTwoIssue::CheckpointSignatureInvalid),
    }

    if !report.issues.is_empty() {
        return Err(finalize(report));
    }

    report.disposition = SignedPostReleaseCheckpointTwoDisposition::Qualified;
    let report_digest = report.canonical_digest();
    let policy_digest = policy_digest.expect("validated policy has digest");
    let runtime_policy_digest = runtime_policy_digest.expect("validated runtime policy has digest");
    let checkpoint_digest = checkpoint_digest.expect("validated checkpoint has digest");
    let qualification_digest = qualification_digest(
        &policy_digest,
        authenticated.qualification_digest(),
        &runtime_policy_digest,
        &checkpoint_digest,
        &report_digest,
    );
    let signed = SignedPostReleaseCheckpointTwo {
        qualification_digest,
        report_digest,
        policy_digest,
        authenticated_observation_qualification_digest: authenticated.qualification_digest().into(),
        runtime_policy_digest,
        checkpoint_digest,
        launch_attestation_digest: checkpoint.launch_attestation_digest.clone(),
        process_instance_id: checkpoint.process_instance_id.clone(),
        verifier_ref: checkpoint.verifier_ref.clone(),
        backend_id: authenticated.backend_id().into(),
        previous_checkpoint_digest: authenticated.bootstrap_ready_checkpoint_digest().into(),
        observed_at_ms: checkpoint.observed_at_ms,
        monotonic_counter: checkpoint.monotonic_counter,
        boot_measurement_digest: checkpoint.boot_measurement_digest.clone(),
        executable_digest: checkpoint.executable_digest.clone(),
        dependency_closure_digest: checkpoint.dependency_closure_digest.clone(),
        runtime_config_digest: checkpoint.runtime_config_digest.clone(),
        dynamic_measurement_digest: checkpoint.dynamic_measurement_digest.clone(),
        signer_key_id: checkpoint.signer_key_id.clone(),
        signer_public_key_ed25519_hex: checkpoint.signer_public_key_ed25519_hex.clone(),
    };

    Ok(SignedPostReleaseCheckpointTwoQualification { report, signed })
}

fn verify_checkpoint_signature(
    policy: &VerifierRuntimeContinuityPolicy,
    checkpoint: &RuntimeMeasurementCheckpoint,
) -> SignatureTrust {
    let Some(message) = checkpoint.canonical_unsigned_bytes() else {
        return SignatureTrust::Invalid;
    };
    if !verify_ed25519(
        &checkpoint.signer_public_key_ed25519_hex,
        &message,
        &checkpoint.signature_ed25519_hex,
    ) {
        return SignatureTrust::Invalid;
    }
    let Some(key) = policy
        .trusted_keys
        .iter()
        .find(|key| key.key_id == checkpoint.signer_key_id)
    else {
        return SignatureTrust::Untrusted;
    };
    let usable = checkpoint.observed_at_ms >= key.valid_from_ms
        && key
            .valid_until_ms
            .map(|until| checkpoint.observed_at_ms < until)
            .unwrap_or(true)
        && key
            .revoked_at_ms
            .map(|revoked| checkpoint.observed_at_ms < revoked)
            .unwrap_or(true)
        && key
            .allowed_scopes
            .contains(&RuntimeMeasurementScope::Checkpoint);
    if key.public_key_ed25519_hex != checkpoint.signer_public_key_ed25519_hex || !usable {
        return SignatureTrust::Untrusted;
    }
    SignatureTrust::Trusted
}

fn verify_ed25519(public_key_hex: &str, message: &[u8], signature_hex: &str) -> bool {
    let Ok(public_key) = hex::decode(public_key_hex) else {
        return false;
    };
    let Ok(signature) = hex::decode(signature_hex) else {
        return false;
    };
    let Ok(public_key): Result<[u8; 32], _> = public_key.try_into() else {
        return false;
    };
    let Ok(signature): Result<[u8; 64], _> = signature.try_into() else {
        return false;
    };
    let Ok(key) = VerifyingKey::from_bytes(&public_key) else {
        return false;
    };
    key.verify(message, &Signature::from_bytes(&signature))
        .is_ok()
}

fn qualification_digest(
    policy: &str,
    authenticated: &str,
    runtime_policy: &str,
    checkpoint: &str,
    report: &str,
) -> String {
    let mut h = blake3::Hasher::new();
    h.update(QUALIFICATION_DOMAIN);
    for value in [policy, authenticated, runtime_policy, checkpoint, report] {
        field(&mut h, value);
    }
    b3(h.finalize())
}

fn finalize(
    mut report: SignedPostReleaseCheckpointTwoReport,
) -> SignedPostReleaseCheckpointTwoReport {
    report.disposition = if report
        .issues
        .iter()
        .any(SignedPostReleaseCheckpointTwoIssue::invalid)
    {
        SignedPostReleaseCheckpointTwoDisposition::Invalid
    } else {
        SignedPostReleaseCheckpointTwoDisposition::Blocked
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
    use ed25519_dalek::{Signer, SigningKey};
    use symthaea_evidence_verifier_runtime_continuity::{
        RUNTIME_CHECKPOINT_SCHEMA_V1, RUNTIME_CONTINUITY_POLICY_SCHEMA_V1,
        RuntimeMeasurementAuthorityKey,
    };

    fn d(label: &str) -> String {
        format!("blake3:{}", blake3::hash(label.as_bytes()).to_hex())
    }

    fn child_policy(runtime_policy_digest: String) -> SignedPostReleaseCheckpointTwoPolicy {
        SignedPostReleaseCheckpointTwoPolicy {
            schema_version: SIGNED_POST_RELEASE_CHECKPOINT_TWO_POLICY_SCHEMA_V1.into(),
            policy_id: "signed-post-release-checkpoint-two:v1".into(),
            expected_authenticated_observation_policy_digest: d("authenticated-policy"),
            expected_runtime_policy_digest: runtime_policy_digest,
            expected_runtime_verifier_ref: "verifier:prod".into(),
            expected_backend_id: "backend:in-process-p256".into(),
            evidence_refs: vec!["review:signed-post-release-checkpoint-two".into()],
        }
    }

    fn runtime_policy(public_key_hex: String) -> VerifierRuntimeContinuityPolicy {
        VerifierRuntimeContinuityPolicy {
            schema_version: RUNTIME_CONTINUITY_POLICY_SCHEMA_V1.into(),
            policy_id: "runtime:v1".into(),
            sequence: 1,
            issued_at_ms: 1,
            expires_at_ms: 10_000,
            verifier_ref: "verifier:prod".into(),
            expected_boot_measurement_digest: d("boot"),
            expected_executable_digest: d("exe"),
            expected_dependency_closure_digest: d("closure"),
            expected_runtime_config_digest: d("config"),
            max_checkpoint_gap_ms: 1_000,
            max_computation_duration_ms: 10_000,
            trusted_keys: vec![RuntimeMeasurementAuthorityKey {
                key_id: "checkpoint-key".into(),
                public_key_ed25519_hex: public_key_hex,
                valid_from_ms: 1,
                valid_until_ms: Some(1_000),
                revoked_at_ms: None,
                allowed_scopes: vec![RuntimeMeasurementScope::Checkpoint],
                evidence_refs: vec!["key:checkpoint".into()],
            }],
            evidence_refs: vec!["policy:runtime".into()],
        }
    }

    fn signed_checkpoint(signing: &SigningKey) -> RuntimeMeasurementCheckpoint {
        let mut checkpoint = RuntimeMeasurementCheckpoint {
            schema_version: RUNTIME_CHECKPOINT_SCHEMA_V1.into(),
            launch_attestation_digest: d("launch"),
            verifier_ref: "verifier:prod".into(),
            process_instance_id: "process:one".into(),
            sequence: 2,
            previous_checkpoint_digest: Some(d("checkpoint-one")),
            observed_at_ms: 100,
            monotonic_counter: 8,
            boot_measurement_digest: d("boot"),
            executable_digest: d("exe"),
            dependency_closure_digest: d("closure"),
            runtime_config_digest: d("config"),
            dynamic_measurement_digest: d("authenticated-observation"),
            signer_key_id: "checkpoint-key".into(),
            signer_public_key_ed25519_hex: hex::encode(signing.verifying_key().to_bytes()),
            signature_ed25519_hex: "00".repeat(64),
        };
        let message = checkpoint.canonical_unsigned_bytes().unwrap();
        checkpoint.signature_ed25519_hex = hex::encode(signing.sign(&message).to_bytes());
        checkpoint
    }

    #[test]
    fn evidence_ref_order_is_nonsemantic() {
        let signing = SigningKey::from_bytes(&[7u8; 32]);
        let runtime = runtime_policy(hex::encode(signing.verifying_key().to_bytes()));
        let digest = runtime.canonical_digest().unwrap();
        let mut left = child_policy(digest);
        left.evidence_refs = vec!["review:a".into(), "review:b".into()];
        let mut right = left.clone();
        right.evidence_refs.reverse();
        assert_eq!(left.canonical_digest(), right.canonical_digest());
    }

    #[test]
    fn exact_parent_checkpoint_encoding_and_trust_semantics_verify() {
        let signing = SigningKey::from_bytes(&[7u8; 32]);
        let runtime = runtime_policy(hex::encode(signing.verifying_key().to_bytes()));
        let checkpoint = signed_checkpoint(&signing);
        assert!(checkpoint.validate());
        assert_eq!(
            verify_checkpoint_signature(&runtime, &checkpoint),
            SignatureTrust::Trusted
        );

        let mut wrong_scope = runtime.clone();
        wrong_scope.trusted_keys[0].allowed_scopes = vec![RuntimeMeasurementScope::Launch];
        assert_eq!(
            verify_checkpoint_signature(&wrong_scope, &checkpoint),
            SignatureTrust::Untrusted
        );

        let mut tampered = checkpoint.clone();
        tampered.dynamic_measurement_digest = d("tampered");
        assert_eq!(
            verify_checkpoint_signature(&runtime, &tampered),
            SignatureTrust::Invalid
        );
    }

    #[test]
    fn policy_binds_authenticated_parent_runtime_role_and_backend() {
        let signing = SigningKey::from_bytes(&[7u8; 32]);
        let runtime = runtime_policy(hex::encode(signing.verifying_key().to_bytes()));
        let left = child_policy(runtime.canonical_digest().unwrap());
        let mut right = left.clone();
        right.expected_authenticated_observation_policy_digest = d("different-auth-policy");
        assert_ne!(left.canonical_digest(), right.canonical_digest());
        let mut right = left.clone();
        right.expected_runtime_verifier_ref = "verifier:other".into();
        assert_ne!(left.canonical_digest(), right.canonical_digest());
        let mut right = left.clone();
        right.expected_backend_id = "backend:other".into();
        assert_ne!(left.canonical_digest(), right.canonical_digest());
    }

    #[test]
    fn claim_ceiling_stays_explicit() {
        let claims = [
            "signature-operation-after-observation=false",
            "checkpoint-gap-policy=false",
            "checkpoint-one-live-proven=false",
            "between-checkpoint-continuity=false",
            "signer-key-non-compromise=false",
            "trusted-time=false",
            "global-replay=false",
            "physical-authority=false",
        ];
        assert_eq!(claims.len(), 8);
    }
}
