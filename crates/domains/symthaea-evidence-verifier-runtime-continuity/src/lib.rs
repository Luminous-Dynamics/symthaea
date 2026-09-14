// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Measured-launch and signed runtime-lineage assurance for verifier computations.
//!
//! This crate deliberately separates:
//! - approved launch identity;
//! - signed, hash-linked runtime measurement continuity; and
//! - one exact computation input/output bound to that lineage.
//!
//! A successful result is bounded evidence about one reviewed measurement model.
//! It does not establish that a software-visible measurement proves a particular
//! TPM/TEE certification level, that the measurement authority is uncompromised,
//! or that supplied timestamps are intrinsically trustworthy.

#![deny(unsafe_code)]

use std::collections::BTreeSet;

use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};

pub const RUNTIME_CONTINUITY_POLICY_SCHEMA_V1: &str =
    "symthaea.assurance.verifier-runtime-continuity-policy.v1";
pub const RUNTIME_LAUNCH_ATTESTATION_SCHEMA_V1: &str =
    "symthaea.assurance.verifier-runtime-launch-attestation.v1";
pub const RUNTIME_CHECKPOINT_SCHEMA_V1: &str =
    "symthaea.assurance.verifier-runtime-checkpoint.v1";
pub const VERIFIER_COMPUTATION_ATTESTATION_SCHEMA_V1: &str =
    "symthaea.assurance.verifier-computation-attestation.v1";
pub const VERIFIER_RUNTIME_EVIDENCE_SCHEMA_V1: &str =
    "symthaea.assurance.verifier-runtime-evidence.v1";
pub const MAX_RUNTIME_KEYS: usize = 1_024;
pub const MAX_RUNTIME_CHECKPOINTS: usize = 65_536;
pub const MAX_EVIDENCE_REFS: usize = 128;
pub const MAX_TEXT_BYTES: usize = 512;

const POLICY_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.verifier-runtime-continuity-policy.digest.v1\0";
const LAUNCH_MESSAGE_DOMAIN: &[u8] =
    b"symthaea.assurance.verifier-runtime-launch.message.v1\0";
const LAUNCH_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.verifier-runtime-launch.digest.v1\0";
const CHECKPOINT_MESSAGE_DOMAIN: &[u8] =
    b"symthaea.assurance.verifier-runtime-checkpoint.message.v1\0";
const CHECKPOINT_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.verifier-runtime-checkpoint.digest.v1\0";
const COMPUTATION_MESSAGE_DOMAIN: &[u8] =
    b"symthaea.assurance.verifier-computation.message.v1\0";
const COMPUTATION_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.verifier-computation.digest.v1\0";
const TRACE_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.verifier-runtime-trace.digest.v1\0";
const CONTINUITY_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.continuous-verifier-execution.digest.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum RuntimeMeasurementScope {
    Launch,
    Checkpoint,
    Computation,
}

impl RuntimeMeasurementScope {
    fn code(self) -> &'static str {
        match self {
            Self::Launch => "launch",
            Self::Checkpoint => "checkpoint",
            Self::Computation => "computation",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeMeasurementAuthorityKey {
    pub key_id: String,
    pub public_key_ed25519_hex: String,
    pub valid_from_ms: u64,
    pub valid_until_ms: Option<u64>,
    pub revoked_at_ms: Option<u64>,
    pub allowed_scopes: Vec<RuntimeMeasurementScope>,
    pub evidence_refs: Vec<String>,
}

impl RuntimeMeasurementAuthorityKey {
    fn validate(&self) -> bool {
        canonical_text(&self.key_id)
            && lower_hex_exact(&self.public_key_ed25519_hex, 64)
            && self
                .valid_until_ms
                .map(|until| until > self.valid_from_ms)
                .unwrap_or(true)
            && self
                .revoked_at_ms
                .map(|revoked| revoked >= self.valid_from_ms)
                .unwrap_or(true)
            && !self.allowed_scopes.is_empty()
            && unique(&self.allowed_scopes)
            && valid_refs(&self.evidence_refs)
    }

    fn usable_for(&self, scope: RuntimeMeasurementScope, at_ms: u64) -> bool {
        at_ms >= self.valid_from_ms
            && self
                .valid_until_ms
                .map(|until| at_ms < until)
                .unwrap_or(true)
            && self.revoked_at_ms.map(|at| at_ms < at).unwrap_or(true)
            && self.allowed_scopes.contains(&scope)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifierRuntimeContinuityPolicy {
    pub schema_version: String,
    pub policy_id: String,
    pub sequence: u64,
    pub issued_at_ms: u64,
    pub expires_at_ms: u64,
    pub verifier_ref: String,
    pub expected_boot_measurement_digest: String,
    pub expected_executable_digest: String,
    pub expected_dependency_closure_digest: String,
    pub expected_runtime_config_digest: String,
    pub max_checkpoint_gap_ms: u64,
    pub max_computation_duration_ms: u64,
    pub trusted_keys: Vec<RuntimeMeasurementAuthorityKey>,
    pub evidence_refs: Vec<String>,
}

impl VerifierRuntimeContinuityPolicy {
    pub fn validate(&self) -> bool {
        if self.schema_version != RUNTIME_CONTINUITY_POLICY_SCHEMA_V1
            || !canonical_text(&self.policy_id)
            || self.sequence == 0
            || self.issued_at_ms >= self.expires_at_ms
            || !canonical_text(&self.verifier_ref)
            || !digest_text(&self.expected_boot_measurement_digest)
            || !digest_text(&self.expected_executable_digest)
            || !digest_text(&self.expected_dependency_closure_digest)
            || !digest_text(&self.expected_runtime_config_digest)
            || self.max_checkpoint_gap_ms == 0
            || self.max_computation_duration_ms == 0
            || self.trusted_keys.is_empty()
            || self.trusted_keys.len() > MAX_RUNTIME_KEYS
            || !valid_refs(&self.evidence_refs)
        {
            return false;
        }
        let mut ids = BTreeSet::new();
        self.trusted_keys
            .iter()
            .all(|key| key.validate() && ids.insert(key.key_id.as_str()))
    }

    pub fn canonical_digest(&self) -> Option<String> {
        if !self.validate() {
            return None;
        }
        let mut hasher = blake3::Hasher::new();
        hasher.update(POLICY_DIGEST_DOMAIN);
        push_field(&mut hasher, &self.schema_version);
        push_field(&mut hasher, &self.policy_id);
        push_u64(&mut hasher, self.sequence);
        push_u64(&mut hasher, self.issued_at_ms);
        push_u64(&mut hasher, self.expires_at_ms);
        push_field(&mut hasher, &self.verifier_ref);
        push_field(&mut hasher, &self.expected_boot_measurement_digest);
        push_field(&mut hasher, &self.expected_executable_digest);
        push_field(&mut hasher, &self.expected_dependency_closure_digest);
        push_field(&mut hasher, &self.expected_runtime_config_digest);
        push_u64(&mut hasher, self.max_checkpoint_gap_ms);
        push_u64(&mut hasher, self.max_computation_duration_ms);

        let mut keys = self.trusted_keys.iter().collect::<Vec<_>>();
        keys.sort_by(|left, right| left.key_id.cmp(&right.key_id));
        for key in keys {
            push_field(&mut hasher, &key.key_id);
            push_field(&mut hasher, &key.public_key_ed25519_hex);
            push_u64(&mut hasher, key.valid_from_ms);
            push_optional_u64(&mut hasher, key.valid_until_ms);
            push_optional_u64(&mut hasher, key.revoked_at_ms);
            let mut scopes = key.allowed_scopes.clone();
            scopes.sort();
            for scope in scopes {
                push_field(&mut hasher, scope.code());
            }
            push_sorted_refs(&mut hasher, &key.evidence_refs);
        }
        push_sorted_refs(&mut hasher, &self.evidence_refs);
        Some(format!("blake3:{}", hasher.finalize().to_hex()))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifierRuntimeLaunchAttestation {
    pub schema_version: String,
    pub verifier_ref: String,
    pub process_instance_id: String,
    pub boot_session_digest: String,
    pub launch_counter: u64,
    pub policy_digest: String,
    pub boot_measurement_digest: String,
    pub executable_digest: String,
    pub dependency_closure_digest: String,
    pub runtime_config_digest: String,
    pub launched_at_ms: u64,
    pub nonce_blake3_hex: String,
    pub signer_key_id: String,
    pub signer_public_key_ed25519_hex: String,
    pub signature_ed25519_hex: String,
}

impl VerifierRuntimeLaunchAttestation {
    pub fn validate(&self) -> bool {
        self.schema_version == RUNTIME_LAUNCH_ATTESTATION_SCHEMA_V1
            && canonical_text(&self.verifier_ref)
            && canonical_text(&self.process_instance_id)
            && digest_text(&self.boot_session_digest)
            && self.launch_counter > 0
            && digest_text(&self.policy_digest)
            && digest_text(&self.boot_measurement_digest)
            && digest_text(&self.executable_digest)
            && digest_text(&self.dependency_closure_digest)
            && digest_text(&self.runtime_config_digest)
            && lower_hex_exact(&self.nonce_blake3_hex, 64)
            && canonical_text(&self.signer_key_id)
            && lower_hex_exact(&self.signer_public_key_ed25519_hex, 64)
            && lower_hex_exact(&self.signature_ed25519_hex, 128)
    }

    pub fn canonical_unsigned_bytes(&self) -> Option<Vec<u8>> {
        if !self.validate() {
            return None;
        }
        let mut bytes = Vec::new();
        bytes.extend_from_slice(LAUNCH_MESSAGE_DOMAIN);
        push_vec_field(&mut bytes, &self.schema_version);
        push_vec_field(&mut bytes, &self.verifier_ref);
        push_vec_field(&mut bytes, &self.process_instance_id);
        push_vec_field(&mut bytes, &self.boot_session_digest);
        bytes.extend_from_slice(&self.launch_counter.to_be_bytes());
        push_vec_field(&mut bytes, &self.policy_digest);
        push_vec_field(&mut bytes, &self.boot_measurement_digest);
        push_vec_field(&mut bytes, &self.executable_digest);
        push_vec_field(&mut bytes, &self.dependency_closure_digest);
        push_vec_field(&mut bytes, &self.runtime_config_digest);
        bytes.extend_from_slice(&self.launched_at_ms.to_be_bytes());
        push_vec_field(&mut bytes, &self.nonce_blake3_hex);
        push_vec_field(&mut bytes, &self.signer_key_id);
        push_vec_field(&mut bytes, &self.signer_public_key_ed25519_hex);
        Some(bytes)
    }

    pub fn canonical_digest(&self) -> Option<String> {
        signed_digest(
            LAUNCH_DIGEST_DOMAIN,
            self.canonical_unsigned_bytes()?,
            &self.signature_ed25519_hex,
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeMeasurementCheckpoint {
    pub schema_version: String,
    pub launch_attestation_digest: String,
    pub verifier_ref: String,
    pub process_instance_id: String,
    pub sequence: u64,
    pub previous_checkpoint_digest: Option<String>,
    pub observed_at_ms: u64,
    pub monotonic_counter: u64,
    pub boot_measurement_digest: String,
    pub executable_digest: String,
    pub dependency_closure_digest: String,
    pub runtime_config_digest: String,
    pub dynamic_measurement_digest: String,
    pub signer_key_id: String,
    pub signer_public_key_ed25519_hex: String,
    pub signature_ed25519_hex: String,
}

impl RuntimeMeasurementCheckpoint {
    pub fn validate(&self) -> bool {
        self.schema_version == RUNTIME_CHECKPOINT_SCHEMA_V1
            && digest_text(&self.launch_attestation_digest)
            && canonical_text(&self.verifier_ref)
            && canonical_text(&self.process_instance_id)
            && self.sequence > 0
            && self
                .previous_checkpoint_digest
                .as_ref()
                .map(|digest| digest_text(digest))
                .unwrap_or(true)
            && self.monotonic_counter > 0
            && digest_text(&self.boot_measurement_digest)
            && digest_text(&self.executable_digest)
            && digest_text(&self.dependency_closure_digest)
            && digest_text(&self.runtime_config_digest)
            && digest_text(&self.dynamic_measurement_digest)
            && canonical_text(&self.signer_key_id)
            && lower_hex_exact(&self.signer_public_key_ed25519_hex, 64)
            && lower_hex_exact(&self.signature_ed25519_hex, 128)
    }

    pub fn canonical_unsigned_bytes(&self) -> Option<Vec<u8>> {
        if !self.validate() {
            return None;
        }
        let mut bytes = Vec::new();
        bytes.extend_from_slice(CHECKPOINT_MESSAGE_DOMAIN);
        push_vec_field(&mut bytes, &self.schema_version);
        push_vec_field(&mut bytes, &self.launch_attestation_digest);
        push_vec_field(&mut bytes, &self.verifier_ref);
        push_vec_field(&mut bytes, &self.process_instance_id);
        bytes.extend_from_slice(&self.sequence.to_be_bytes());
        push_vec_optional_string(&mut bytes, self.previous_checkpoint_digest.as_deref());
        bytes.extend_from_slice(&self.observed_at_ms.to_be_bytes());
        bytes.extend_from_slice(&self.monotonic_counter.to_be_bytes());
        push_vec_field(&mut bytes, &self.boot_measurement_digest);
        push_vec_field(&mut bytes, &self.executable_digest);
        push_vec_field(&mut bytes, &self.dependency_closure_digest);
        push_vec_field(&mut bytes, &self.runtime_config_digest);
        push_vec_field(&mut bytes, &self.dynamic_measurement_digest);
        push_vec_field(&mut bytes, &self.signer_key_id);
        push_vec_field(&mut bytes, &self.signer_public_key_ed25519_hex);
        Some(bytes)
    }

    pub fn canonical_digest(&self) -> Option<String> {
        signed_digest(
            CHECKPOINT_DIGEST_DOMAIN,
            self.canonical_unsigned_bytes()?,
            &self.signature_ed25519_hex,
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifierComputationAttestation {
    pub schema_version: String,
    pub launch_attestation_digest: String,
    pub final_checkpoint_digest: String,
    pub verifier_ref: String,
    pub process_instance_id: String,
    pub request_nonce_blake3_hex: String,
    pub input_digest: String,
    pub output_digest: String,
    pub started_at_ms: u64,
    pub completed_at_ms: u64,
    pub attested_at_ms: u64,
    pub computation_counter: u64,
    pub signer_key_id: String,
    pub signer_public_key_ed25519_hex: String,
    pub signature_ed25519_hex: String,
}

impl VerifierComputationAttestation {
    pub fn validate(&self) -> bool {
        self.schema_version == VERIFIER_COMPUTATION_ATTESTATION_SCHEMA_V1
            && digest_text(&self.launch_attestation_digest)
            && digest_text(&self.final_checkpoint_digest)
            && canonical_text(&self.verifier_ref)
            && canonical_text(&self.process_instance_id)
            && lower_hex_exact(&self.request_nonce_blake3_hex, 64)
            && digest_text(&self.input_digest)
            && digest_text(&self.output_digest)
            && self.started_at_ms <= self.completed_at_ms
            && self.completed_at_ms <= self.attested_at_ms
            && self.computation_counter > 0
            && canonical_text(&self.signer_key_id)
            && lower_hex_exact(&self.signer_public_key_ed25519_hex, 64)
            && lower_hex_exact(&self.signature_ed25519_hex, 128)
    }

    pub fn canonical_unsigned_bytes(&self) -> Option<Vec<u8>> {
        if !self.validate() {
            return None;
        }
        let mut bytes = Vec::new();
        bytes.extend_from_slice(COMPUTATION_MESSAGE_DOMAIN);
        push_vec_field(&mut bytes, &self.schema_version);
        push_vec_field(&mut bytes, &self.launch_attestation_digest);
        push_vec_field(&mut bytes, &self.final_checkpoint_digest);
        push_vec_field(&mut bytes, &self.verifier_ref);
        push_vec_field(&mut bytes, &self.process_instance_id);
        push_vec_field(&mut bytes, &self.request_nonce_blake3_hex);
        push_vec_field(&mut bytes, &self.input_digest);
        push_vec_field(&mut bytes, &self.output_digest);
        bytes.extend_from_slice(&self.started_at_ms.to_be_bytes());
        bytes.extend_from_slice(&self.completed_at_ms.to_be_bytes());
        bytes.extend_from_slice(&self.attested_at_ms.to_be_bytes());
        bytes.extend_from_slice(&self.computation_counter.to_be_bytes());
        push_vec_field(&mut bytes, &self.signer_key_id);
        push_vec_field(&mut bytes, &self.signer_public_key_ed25519_hex);
        Some(bytes)
    }

    pub fn canonical_digest(&self) -> Option<String> {
        signed_digest(
            COMPUTATION_DIGEST_DOMAIN,
            self.canonical_unsigned_bytes()?,
            &self.signature_ed25519_hex,
        )
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifierRuntimeEvidence {
    pub schema_version: String,
    pub launch: VerifierRuntimeLaunchAttestation,
    pub checkpoints: Vec<RuntimeMeasurementCheckpoint>,
    pub computation: VerifierComputationAttestation,
    pub evidence_refs: Vec<String>,
}

impl VerifierRuntimeEvidence {
    pub fn validate(&self) -> bool {
        self.schema_version == VERIFIER_RUNTIME_EVIDENCE_SCHEMA_V1
            && self.launch.validate()
            && !self.checkpoints.is_empty()
            && self.checkpoints.len() <= MAX_RUNTIME_CHECKPOINTS
            && self.checkpoints.iter().all(RuntimeMeasurementCheckpoint::validate)
            && self.computation.validate()
            && valid_refs(&self.evidence_refs)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RuntimeContinuityDisposition {
    Invalid,
    Blocked,
    Qualified,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum RuntimeContinuityIssue {
    InvalidPolicy,
    InvalidEvidence,
    PolicyDigestMismatch,
    VerifierMismatch,
    ProcessInstanceMismatch,
    LaunchSignatureInvalid,
    LaunchSignerUntrusted,
    LaunchOutsidePolicyWindow,
    BootMeasurementMismatch,
    ExecutableDigestMismatch,
    DependencyClosureDigestMismatch,
    RuntimeConfigDigestMismatch,
    CheckpointSequenceMismatch { expected: u64, observed: u64 },
    CheckpointLinkMismatch { sequence: u64 },
    CheckpointTimeRegression { sequence: u64 },
    CheckpointGapExceeded { sequence: u64, gap_ms: u64, maximum_ms: u64 },
    CheckpointCounterNotMonotonic { sequence: u64 },
    CheckpointSignatureInvalid { sequence: u64 },
    CheckpointSignerUntrusted { sequence: u64 },
    CheckpointStaticIdentityDrift { sequence: u64 },
    ComputationLaunchMismatch,
    ComputationFinalCheckpointMismatch,
    ComputationSignatureInvalid,
    ComputationSignerUntrusted,
    ComputationStartsBeforeLaunch,
    ComputationDurationExceeded { observed_ms: u64, maximum_ms: u64 },
    FinalCheckpointPrecedesComputationCompletion,
    ComputationCounterNotAfterRuntimeCounter,
    EvidenceFromFuture,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RuntimeContinuityReport {
    pub disposition: RuntimeContinuityDisposition,
    pub verifier_ref: String,
    pub process_instance_id: String,
    pub policy_digest: Option<String>,
    pub launch_attestation_digest: Option<String>,
    pub final_checkpoint_digest: Option<String>,
    pub computation_attestation_digest: Option<String>,
    pub runtime_trace_digest: Option<String>,
    pub input_digest: String,
    pub output_digest: String,
    pub request_nonce_blake3_hex: String,
    pub started_at_ms: u64,
    pub completed_at_ms: u64,
    pub assessed_at_ms: u64,
    pub issues: Vec<RuntimeContinuityIssue>,
    pub continuity_digest: Option<String>,
}

impl RuntimeContinuityReport {
    pub const fn grants_physical_authority(&self) -> bool {
        false
    }

    pub const fn hardware_certification_established(&self) -> bool {
        false
    }

    pub const fn trusted_time_established(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ContinuousVerifierExecution {
    continuity_digest: String,
    verifier_ref: String,
    process_instance_id: String,
    policy_digest: String,
    launch_attestation_digest: String,
    final_checkpoint_digest: String,
    computation_attestation_digest: String,
    runtime_trace_digest: String,
    request_nonce_blake3_hex: String,
    input_digest: String,
    output_digest: String,
    started_at_ms: u64,
    completed_at_ms: u64,
    assessed_at_ms: u64,
}

impl ContinuousVerifierExecution {
    pub fn continuity_digest(&self) -> &str { &self.continuity_digest }
    pub fn verifier_ref(&self) -> &str { &self.verifier_ref }
    pub fn process_instance_id(&self) -> &str { &self.process_instance_id }
    pub fn policy_digest(&self) -> &str { &self.policy_digest }
    pub fn launch_attestation_digest(&self) -> &str { &self.launch_attestation_digest }
    pub fn final_checkpoint_digest(&self) -> &str { &self.final_checkpoint_digest }
    pub fn computation_attestation_digest(&self) -> &str { &self.computation_attestation_digest }
    pub fn runtime_trace_digest(&self) -> &str { &self.runtime_trace_digest }
    pub fn request_nonce_blake3_hex(&self) -> &str { &self.request_nonce_blake3_hex }
    pub fn input_digest(&self) -> &str { &self.input_digest }
    pub fn output_digest(&self) -> &str { &self.output_digest }
    pub const fn started_at_ms(&self) -> u64 { self.started_at_ms }
    pub const fn completed_at_ms(&self) -> u64 { self.completed_at_ms }
    pub const fn assessed_at_ms(&self) -> u64 { self.assessed_at_ms }
    pub const fn grants_physical_authority(&self) -> bool { false }
    pub const fn hardware_certification_established(&self) -> bool { false }
    pub const fn trusted_time_established(&self) -> bool { false }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeContinuityAssessment {
    pub report: RuntimeContinuityReport,
    continuous: Option<ContinuousVerifierExecution>,
}

impl RuntimeContinuityAssessment {
    pub fn continuous(&self) -> Option<&ContinuousVerifierExecution> {
        self.continuous.as_ref()
    }

    pub fn into_continuous(self) -> Option<ContinuousVerifierExecution> {
        self.continuous
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SignatureTrust {
    Trusted,
    Untrusted,
    Invalid,
}

pub fn verify_continuous_verifier_execution(
    policy: &VerifierRuntimeContinuityPolicy,
    evidence: &VerifierRuntimeEvidence,
    assessed_at_ms: u64,
) -> RuntimeContinuityAssessment {
    let policy_digest = policy.canonical_digest();
    let launch_digest = evidence.launch.canonical_digest();
    let computation_digest = evidence.computation.canonical_digest();
    let final_checkpoint_digest = evidence
        .checkpoints
        .last()
        .and_then(|checkpoint| checkpoint.canonical_digest());
    let mut issues = Vec::new();

    if !policy.validate() {
        issues.push(RuntimeContinuityIssue::InvalidPolicy);
    }
    if !evidence.validate() {
        issues.push(RuntimeContinuityIssue::InvalidEvidence);
    }
    if issues.iter().any(is_structural_issue) {
        return assessment(
            policy,
            evidence,
            assessed_at_ms,
            policy_digest,
            launch_digest,
            final_checkpoint_digest,
            computation_digest,
            None,
            issues,
            None,
        );
    }

    let policy_digest_value = policy_digest.clone().expect("validated policy");
    let launch_digest_value = launch_digest.clone().expect("validated launch");

    if evidence.launch.policy_digest != policy_digest_value {
        issues.push(RuntimeContinuityIssue::PolicyDigestMismatch);
    }
    if evidence.launch.verifier_ref != policy.verifier_ref
        || evidence.computation.verifier_ref != policy.verifier_ref
        || evidence
            .checkpoints
            .iter()
            .any(|checkpoint| checkpoint.verifier_ref != policy.verifier_ref)
    {
        issues.push(RuntimeContinuityIssue::VerifierMismatch);
    }
    if evidence.computation.process_instance_id != evidence.launch.process_instance_id
        || evidence
            .checkpoints
            .iter()
            .any(|checkpoint| checkpoint.process_instance_id != evidence.launch.process_instance_id)
    {
        issues.push(RuntimeContinuityIssue::ProcessInstanceMismatch);
    }

    match verify_signed_scope(
        policy,
        RuntimeMeasurementScope::Launch,
        evidence.launch.launched_at_ms,
        &evidence.launch.signer_key_id,
        &evidence.launch.signer_public_key_ed25519_hex,
        evidence.launch.canonical_unsigned_bytes().as_deref(),
        &evidence.launch.signature_ed25519_hex,
    ) {
        SignatureTrust::Trusted => {}
        SignatureTrust::Untrusted => issues.push(RuntimeContinuityIssue::LaunchSignerUntrusted),
        SignatureTrust::Invalid => issues.push(RuntimeContinuityIssue::LaunchSignatureInvalid),
    }
    if !(policy.issued_at_ms <= evidence.launch.launched_at_ms
        && evidence.launch.launched_at_ms < policy.expires_at_ms)
    {
        issues.push(RuntimeContinuityIssue::LaunchOutsidePolicyWindow);
    }

    if evidence.launch.boot_measurement_digest != policy.expected_boot_measurement_digest {
        issues.push(RuntimeContinuityIssue::BootMeasurementMismatch);
    }
    if evidence.launch.executable_digest != policy.expected_executable_digest {
        issues.push(RuntimeContinuityIssue::ExecutableDigestMismatch);
    }
    if evidence.launch.dependency_closure_digest != policy.expected_dependency_closure_digest {
        issues.push(RuntimeContinuityIssue::DependencyClosureDigestMismatch);
    }
    if evidence.launch.runtime_config_digest != policy.expected_runtime_config_digest {
        issues.push(RuntimeContinuityIssue::RuntimeConfigDigestMismatch);
    }

    let mut previous_digest: Option<String> = None;
    let mut previous_time = evidence.launch.launched_at_ms;
    let mut previous_counter = evidence.launch.launch_counter;
    let mut checkpoint_digests = Vec::with_capacity(evidence.checkpoints.len());

    for (index, checkpoint) in evidence.checkpoints.iter().enumerate() {
        let expected_sequence = index as u64 + 1;
        let digest = checkpoint.canonical_digest().expect("validated checkpoint");
        checkpoint_digests.push(digest.clone());

        if checkpoint.sequence != expected_sequence {
            issues.push(RuntimeContinuityIssue::CheckpointSequenceMismatch {
                expected: expected_sequence,
                observed: checkpoint.sequence,
            });
        }
        if checkpoint.launch_attestation_digest != launch_digest_value {
            issues.push(RuntimeContinuityIssue::CheckpointLinkMismatch {
                sequence: checkpoint.sequence,
            });
        }
        let expected_previous = if expected_sequence == 1 {
            None
        } else {
            previous_digest.as_deref()
        };
        if checkpoint.previous_checkpoint_digest.as_deref() != expected_previous {
            issues.push(RuntimeContinuityIssue::CheckpointLinkMismatch {
                sequence: checkpoint.sequence,
            });
        }
        if checkpoint.observed_at_ms < previous_time {
            issues.push(RuntimeContinuityIssue::CheckpointTimeRegression {
                sequence: checkpoint.sequence,
            });
        } else {
            let gap = checkpoint.observed_at_ms - previous_time;
            if gap > policy.max_checkpoint_gap_ms {
                issues.push(RuntimeContinuityIssue::CheckpointGapExceeded {
                    sequence: checkpoint.sequence,
                    gap_ms: gap,
                    maximum_ms: policy.max_checkpoint_gap_ms,
                });
            }
        }
        if checkpoint.monotonic_counter <= previous_counter {
            issues.push(RuntimeContinuityIssue::CheckpointCounterNotMonotonic {
                sequence: checkpoint.sequence,
            });
        }
        if checkpoint.boot_measurement_digest != evidence.launch.boot_measurement_digest
            || checkpoint.executable_digest != evidence.launch.executable_digest
            || checkpoint.dependency_closure_digest != evidence.launch.dependency_closure_digest
            || checkpoint.runtime_config_digest != evidence.launch.runtime_config_digest
        {
            issues.push(RuntimeContinuityIssue::CheckpointStaticIdentityDrift {
                sequence: checkpoint.sequence,
            });
        }
        match verify_signed_scope(
            policy,
            RuntimeMeasurementScope::Checkpoint,
            checkpoint.observed_at_ms,
            &checkpoint.signer_key_id,
            &checkpoint.signer_public_key_ed25519_hex,
            checkpoint.canonical_unsigned_bytes().as_deref(),
            &checkpoint.signature_ed25519_hex,
        ) {
            SignatureTrust::Trusted => {}
            SignatureTrust::Untrusted => issues.push(
                RuntimeContinuityIssue::CheckpointSignerUntrusted {
                    sequence: checkpoint.sequence,
                },
            ),
            SignatureTrust::Invalid => issues.push(
                RuntimeContinuityIssue::CheckpointSignatureInvalid {
                    sequence: checkpoint.sequence,
                },
            ),
        }
        previous_digest = Some(digest);
        previous_time = checkpoint.observed_at_ms;
        previous_counter = checkpoint.monotonic_counter;
    }

    let final_checkpoint_digest_value = final_checkpoint_digest
        .clone()
        .expect("validated non-empty checkpoint list");
    let computation = &evidence.computation;
    if computation.launch_attestation_digest != launch_digest_value {
        issues.push(RuntimeContinuityIssue::ComputationLaunchMismatch);
    }
    if computation.final_checkpoint_digest != final_checkpoint_digest_value {
        issues.push(RuntimeContinuityIssue::ComputationFinalCheckpointMismatch);
    }
    if computation.started_at_ms < evidence.launch.launched_at_ms {
        issues.push(RuntimeContinuityIssue::ComputationStartsBeforeLaunch);
    }
    let duration = computation.completed_at_ms.saturating_sub(computation.started_at_ms);
    if duration > policy.max_computation_duration_ms {
        issues.push(RuntimeContinuityIssue::ComputationDurationExceeded {
            observed_ms: duration,
            maximum_ms: policy.max_computation_duration_ms,
        });
    }
    if previous_time < computation.completed_at_ms {
        issues.push(RuntimeContinuityIssue::FinalCheckpointPrecedesComputationCompletion);
    }
    if computation.computation_counter <= previous_counter {
        issues.push(RuntimeContinuityIssue::ComputationCounterNotAfterRuntimeCounter);
    }
    match verify_signed_scope(
        policy,
        RuntimeMeasurementScope::Computation,
        computation.attested_at_ms,
        &computation.signer_key_id,
        &computation.signer_public_key_ed25519_hex,
        computation.canonical_unsigned_bytes().as_deref(),
        &computation.signature_ed25519_hex,
    ) {
        SignatureTrust::Trusted => {}
        SignatureTrust::Untrusted => issues.push(RuntimeContinuityIssue::ComputationSignerUntrusted),
        SignatureTrust::Invalid => issues.push(RuntimeContinuityIssue::ComputationSignatureInvalid),
    }

    if evidence.launch.launched_at_ms > assessed_at_ms
        || evidence
            .checkpoints
            .iter()
            .any(|checkpoint| checkpoint.observed_at_ms > assessed_at_ms)
        || computation.attested_at_ms > assessed_at_ms
    {
        issues.push(RuntimeContinuityIssue::EvidenceFromFuture);
    }

    let trace_digest = runtime_trace_digest(
        &policy_digest_value,
        &launch_digest_value,
        &checkpoint_digests,
        &computation_digest.clone().expect("validated computation"),
        assessed_at_ms,
    );

    let structurally_invalid = issues.iter().any(is_structural_issue);
    let blocked = issues.iter().any(is_blocking_issue);
    let disposition = if structurally_invalid {
        RuntimeContinuityDisposition::Invalid
    } else if blocked {
        RuntimeContinuityDisposition::Blocked
    } else {
        RuntimeContinuityDisposition::Qualified
    };

    let continuous = if disposition == RuntimeContinuityDisposition::Qualified {
        let computation_digest_value = computation_digest.clone().expect("validated computation");
        let continuity_digest = continuity_digest(
            &policy_digest_value,
            &launch_digest_value,
            &final_checkpoint_digest_value,
            &computation_digest_value,
            &trace_digest,
            computation,
            assessed_at_ms,
        );
        Some(ContinuousVerifierExecution {
            continuity_digest,
            verifier_ref: policy.verifier_ref.clone(),
            process_instance_id: evidence.launch.process_instance_id.clone(),
            policy_digest: policy_digest_value,
            launch_attestation_digest: launch_digest_value,
            final_checkpoint_digest: final_checkpoint_digest_value,
            computation_attestation_digest: computation_digest_value,
            runtime_trace_digest: trace_digest.clone(),
            request_nonce_blake3_hex: computation.request_nonce_blake3_hex.clone(),
            input_digest: computation.input_digest.clone(),
            output_digest: computation.output_digest.clone(),
            started_at_ms: computation.started_at_ms,
            completed_at_ms: computation.completed_at_ms,
            assessed_at_ms,
        })
    } else {
        None
    };

    assessment(
        policy,
        evidence,
        assessed_at_ms,
        policy_digest,
        launch_digest,
        final_checkpoint_digest,
        computation_digest,
        Some(trace_digest),
        issues,
        continuous,
    )
}

#[allow(clippy::too_many_arguments)]
fn assessment(
    policy: &VerifierRuntimeContinuityPolicy,
    evidence: &VerifierRuntimeEvidence,
    assessed_at_ms: u64,
    policy_digest: Option<String>,
    launch_digest: Option<String>,
    final_checkpoint_digest: Option<String>,
    computation_digest: Option<String>,
    trace_digest: Option<String>,
    issues: Vec<RuntimeContinuityIssue>,
    continuous: Option<ContinuousVerifierExecution>,
) -> RuntimeContinuityAssessment {
    let disposition = if issues.iter().any(is_structural_issue) {
        RuntimeContinuityDisposition::Invalid
    } else if issues.iter().any(is_blocking_issue) {
        RuntimeContinuityDisposition::Blocked
    } else if continuous.is_some() {
        RuntimeContinuityDisposition::Qualified
    } else {
        RuntimeContinuityDisposition::Invalid
    };
    let continuity_digest = continuous
        .as_ref()
        .map(|value| value.continuity_digest().to_string());
    RuntimeContinuityAssessment {
        report: RuntimeContinuityReport {
            disposition,
            verifier_ref: policy.verifier_ref.clone(),
            process_instance_id: evidence.launch.process_instance_id.clone(),
            policy_digest,
            launch_attestation_digest: launch_digest,
            final_checkpoint_digest,
            computation_attestation_digest: computation_digest,
            runtime_trace_digest: trace_digest,
            input_digest: evidence.computation.input_digest.clone(),
            output_digest: evidence.computation.output_digest.clone(),
            request_nonce_blake3_hex: evidence.computation.request_nonce_blake3_hex.clone(),
            started_at_ms: evidence.computation.started_at_ms,
            completed_at_ms: evidence.computation.completed_at_ms,
            assessed_at_ms,
            issues,
            continuity_digest,
        },
        continuous,
    }
}

fn verify_signed_scope(
    policy: &VerifierRuntimeContinuityPolicy,
    scope: RuntimeMeasurementScope,
    at_ms: u64,
    key_id: &str,
    public_key_hex: &str,
    message: Option<&[u8]>,
    signature_hex: &str,
) -> SignatureTrust {
    let Some(message) = message else {
        return SignatureTrust::Invalid;
    };
    if !verify_ed25519(public_key_hex, message, signature_hex) {
        return SignatureTrust::Invalid;
    }
    let Some(key) = policy.trusted_keys.iter().find(|key| key.key_id == key_id) else {
        return SignatureTrust::Untrusted;
    };
    if key.public_key_ed25519_hex != public_key_hex || !key.usable_for(scope, at_ms) {
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
    key.verify(message, &Signature::from_bytes(&signature)).is_ok()
}

fn runtime_trace_digest(
    policy_digest: &str,
    launch_digest: &str,
    checkpoints: &[String],
    computation_digest: &str,
    assessed_at_ms: u64,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(TRACE_DIGEST_DOMAIN);
    push_field(&mut hasher, policy_digest);
    push_field(&mut hasher, launch_digest);
    for checkpoint in checkpoints {
        push_field(&mut hasher, checkpoint);
    }
    push_field(&mut hasher, computation_digest);
    push_u64(&mut hasher, assessed_at_ms);
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn continuity_digest(
    policy_digest: &str,
    launch_digest: &str,
    final_checkpoint_digest: &str,
    computation_digest: &str,
    trace_digest: &str,
    computation: &VerifierComputationAttestation,
    assessed_at_ms: u64,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(CONTINUITY_DIGEST_DOMAIN);
    for field in [
        policy_digest,
        launch_digest,
        final_checkpoint_digest,
        computation_digest,
        trace_digest,
        computation.request_nonce_blake3_hex.as_str(),
        computation.input_digest.as_str(),
        computation.output_digest.as_str(),
    ] {
        push_field(&mut hasher, field);
    }
    push_u64(&mut hasher, computation.started_at_ms);
    push_u64(&mut hasher, computation.completed_at_ms);
    push_u64(&mut hasher, assessed_at_ms);
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn signed_digest(domain: &[u8], unsigned: Vec<u8>, signature_hex: &str) -> Option<String> {
    if !lower_hex_exact(signature_hex, 128) {
        return None;
    }
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    hasher.update(&(unsigned.len() as u64).to_be_bytes());
    hasher.update(&unsigned);
    push_field(&mut hasher, signature_hex);
    Some(format!("blake3:{}", hasher.finalize().to_hex()))
}

fn is_structural_issue(issue: &RuntimeContinuityIssue) -> bool {
    matches!(
        issue,
        RuntimeContinuityIssue::InvalidPolicy
            | RuntimeContinuityIssue::InvalidEvidence
            | RuntimeContinuityIssue::PolicyDigestMismatch
            | RuntimeContinuityIssue::VerifierMismatch
            | RuntimeContinuityIssue::ProcessInstanceMismatch
            | RuntimeContinuityIssue::LaunchSignatureInvalid
            | RuntimeContinuityIssue::CheckpointSequenceMismatch { .. }
            | RuntimeContinuityIssue::CheckpointLinkMismatch { .. }
            | RuntimeContinuityIssue::CheckpointTimeRegression { .. }
            | RuntimeContinuityIssue::CheckpointCounterNotMonotonic { .. }
            | RuntimeContinuityIssue::CheckpointSignatureInvalid { .. }
            | RuntimeContinuityIssue::ComputationLaunchMismatch
            | RuntimeContinuityIssue::ComputationFinalCheckpointMismatch
            | RuntimeContinuityIssue::ComputationSignatureInvalid
            | RuntimeContinuityIssue::ComputationStartsBeforeLaunch
            | RuntimeContinuityIssue::ComputationCounterNotAfterRuntimeCounter
            | RuntimeContinuityIssue::EvidenceFromFuture
    )
}

fn is_blocking_issue(issue: &RuntimeContinuityIssue) -> bool {
    matches!(
        issue,
        RuntimeContinuityIssue::LaunchSignerUntrusted
            | RuntimeContinuityIssue::LaunchOutsidePolicyWindow
            | RuntimeContinuityIssue::BootMeasurementMismatch
            | RuntimeContinuityIssue::ExecutableDigestMismatch
            | RuntimeContinuityIssue::DependencyClosureDigestMismatch
            | RuntimeContinuityIssue::RuntimeConfigDigestMismatch
            | RuntimeContinuityIssue::CheckpointGapExceeded { .. }
            | RuntimeContinuityIssue::CheckpointSignerUntrusted { .. }
            | RuntimeContinuityIssue::CheckpointStaticIdentityDrift { .. }
            | RuntimeContinuityIssue::ComputationSignerUntrusted
            | RuntimeContinuityIssue::ComputationDurationExceeded { .. }
            | RuntimeContinuityIssue::FinalCheckpointPrecedesComputationCompletion
    )
}

fn canonical_text(value: &str) -> bool {
    !value.is_empty()
        && value == value.trim()
        && value.len() <= MAX_TEXT_BYTES
        && !value.chars().any(char::is_control)
}

fn digest_text(value: &str) -> bool {
    value.strip_prefix("blake3:").is_some_and(|digest| {
        digest.len() == 64
            && digest
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    })
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
        && unique(refs)
}

fn unique<T: Ord + Clone>(values: &[T]) -> bool {
    values.iter().cloned().collect::<BTreeSet<_>>().len() == values.len()
}

fn push_field(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_be_bytes());
    hasher.update(value.as_bytes());
}

fn push_u64(hasher: &mut blake3::Hasher, value: u64) {
    hasher.update(&value.to_be_bytes());
}

fn push_optional_u64(hasher: &mut blake3::Hasher, value: Option<u64>) {
    match value {
        Some(value) => {
            hasher.update(&[1]);
            push_u64(hasher, value);
        }
        None => hasher.update(&[0]),
    }
}

fn push_sorted_refs(hasher: &mut blake3::Hasher, refs: &[String]) {
    let mut refs = refs.to_vec();
    refs.sort();
    for reference in refs {
        push_field(hasher, &reference);
    }
}

fn push_vec_field(bytes: &mut Vec<u8>, value: &str) {
    bytes.extend_from_slice(&(value.len() as u64).to_be_bytes());
    bytes.extend_from_slice(value.as_bytes());
}

fn push_vec_optional_string(bytes: &mut Vec<u8>, value: Option<&str>) {
    match value {
        Some(value) => {
            bytes.push(1);
            push_vec_field(bytes, value);
        }
        None => bytes.push(0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};

    fn d(label: &str) -> String {
        format!("blake3:{}", blake3::hash(label.as_bytes()).to_hex())
    }

    fn signing_key() -> SigningKey {
        SigningKey::from_bytes(&[7u8; 32])
    }

    fn public_key_hex() -> String {
        hex::encode(signing_key().verifying_key().as_bytes())
    }

    fn policy() -> VerifierRuntimeContinuityPolicy {
        VerifierRuntimeContinuityPolicy {
            schema_version: RUNTIME_CONTINUITY_POLICY_SCHEMA_V1.into(),
            policy_id: "runtime-policy:1".into(),
            sequence: 1,
            issued_at_ms: 1_000,
            expires_at_ms: 20_000,
            verifier_ref: "verifier:obligation".into(),
            expected_boot_measurement_digest: d("boot"),
            expected_executable_digest: d("exe"),
            expected_dependency_closure_digest: d("closure"),
            expected_runtime_config_digest: d("config"),
            max_checkpoint_gap_ms: 2_000,
            max_computation_duration_ms: 5_000,
            trusted_keys: vec![RuntimeMeasurementAuthorityKey {
                key_id: "runtime-key:1".into(),
                public_key_ed25519_hex: public_key_hex(),
                valid_from_ms: 500,
                valid_until_ms: Some(19_000),
                revoked_at_ms: None,
                allowed_scopes: vec![
                    RuntimeMeasurementScope::Launch,
                    RuntimeMeasurementScope::Checkpoint,
                    RuntimeMeasurementScope::Computation,
                ],
                evidence_refs: vec!["review:key".into()],
            }],
            evidence_refs: vec!["review:runtime-policy".into()],
        }
    }

    fn sign_launch(mut launch: VerifierRuntimeLaunchAttestation) -> VerifierRuntimeLaunchAttestation {
        let message = launch.canonical_unsigned_bytes().unwrap();
        launch.signature_ed25519_hex = hex::encode(signing_key().sign(&message).to_bytes());
        launch
    }

    fn sign_checkpoint(mut checkpoint: RuntimeMeasurementCheckpoint) -> RuntimeMeasurementCheckpoint {
        let message = checkpoint.canonical_unsigned_bytes().unwrap();
        checkpoint.signature_ed25519_hex = hex::encode(signing_key().sign(&message).to_bytes());
        checkpoint
    }

    fn sign_computation(
        mut computation: VerifierComputationAttestation,
    ) -> VerifierComputationAttestation {
        let message = computation.canonical_unsigned_bytes().unwrap();
        computation.signature_ed25519_hex = hex::encode(signing_key().sign(&message).to_bytes());
        computation
    }

    fn evidence() -> VerifierRuntimeEvidence {
        let policy = policy();
        let launch = sign_launch(VerifierRuntimeLaunchAttestation {
            schema_version: RUNTIME_LAUNCH_ATTESTATION_SCHEMA_V1.into(),
            verifier_ref: policy.verifier_ref.clone(),
            process_instance_id: "process:abc".into(),
            boot_session_digest: d("boot-session"),
            launch_counter: 100,
            policy_digest: policy.canonical_digest().unwrap(),
            boot_measurement_digest: policy.expected_boot_measurement_digest.clone(),
            executable_digest: policy.expected_executable_digest.clone(),
            dependency_closure_digest: policy.expected_dependency_closure_digest.clone(),
            runtime_config_digest: policy.expected_runtime_config_digest.clone(),
            launched_at_ms: 2_000,
            nonce_blake3_hex: "11".repeat(32),
            signer_key_id: "runtime-key:1".into(),
            signer_public_key_ed25519_hex: public_key_hex(),
            signature_ed25519_hex: "00".repeat(64),
        });
        let launch_digest = launch.canonical_digest().unwrap();
        let cp1 = sign_checkpoint(RuntimeMeasurementCheckpoint {
            schema_version: RUNTIME_CHECKPOINT_SCHEMA_V1.into(),
            launch_attestation_digest: launch_digest.clone(),
            verifier_ref: policy.verifier_ref.clone(),
            process_instance_id: launch.process_instance_id.clone(),
            sequence: 1,
            previous_checkpoint_digest: None,
            observed_at_ms: 3_000,
            monotonic_counter: 101,
            boot_measurement_digest: launch.boot_measurement_digest.clone(),
            executable_digest: launch.executable_digest.clone(),
            dependency_closure_digest: launch.dependency_closure_digest.clone(),
            runtime_config_digest: launch.runtime_config_digest.clone(),
            dynamic_measurement_digest: d("dynamic:1"),
            signer_key_id: "runtime-key:1".into(),
            signer_public_key_ed25519_hex: public_key_hex(),
            signature_ed25519_hex: "00".repeat(64),
        });
        let cp1_digest = cp1.canonical_digest().unwrap();
        let cp2 = sign_checkpoint(RuntimeMeasurementCheckpoint {
            schema_version: RUNTIME_CHECKPOINT_SCHEMA_V1.into(),
            launch_attestation_digest: launch_digest.clone(),
            verifier_ref: policy.verifier_ref.clone(),
            process_instance_id: launch.process_instance_id.clone(),
            sequence: 2,
            previous_checkpoint_digest: Some(cp1_digest),
            observed_at_ms: 4_500,
            monotonic_counter: 102,
            boot_measurement_digest: launch.boot_measurement_digest.clone(),
            executable_digest: launch.executable_digest.clone(),
            dependency_closure_digest: launch.dependency_closure_digest.clone(),
            runtime_config_digest: launch.runtime_config_digest.clone(),
            dynamic_measurement_digest: d("dynamic:2"),
            signer_key_id: "runtime-key:1".into(),
            signer_public_key_ed25519_hex: public_key_hex(),
            signature_ed25519_hex: "00".repeat(64),
        });
        let cp2_digest = cp2.canonical_digest().unwrap();
        let computation = sign_computation(VerifierComputationAttestation {
            schema_version: VERIFIER_COMPUTATION_ATTESTATION_SCHEMA_V1.into(),
            launch_attestation_digest: launch_digest,
            final_checkpoint_digest: cp2_digest,
            verifier_ref: policy.verifier_ref.clone(),
            process_instance_id: launch.process_instance_id.clone(),
            request_nonce_blake3_hex: "22".repeat(32),
            input_digest: d("input"),
            output_digest: d("output"),
            started_at_ms: 3_200,
            completed_at_ms: 4_400,
            attested_at_ms: 4_600,
            computation_counter: 103,
            signer_key_id: "runtime-key:1".into(),
            signer_public_key_ed25519_hex: public_key_hex(),
            signature_ed25519_hex: "00".repeat(64),
        });
        VerifierRuntimeEvidence {
            schema_version: VERIFIER_RUNTIME_EVIDENCE_SCHEMA_V1.into(),
            launch,
            checkpoints: vec![cp1, cp2],
            computation,
            evidence_refs: vec!["run:evidence".into()],
        }
    }

    #[test]
    fn continuous_signed_lineage_qualifies() {
        let result = verify_continuous_verifier_execution(&policy(), &evidence(), 5_000);
        assert_eq!(result.report.disposition, RuntimeContinuityDisposition::Qualified);
        let active = result.active_or_continuous_for_test();
        assert_eq!(active.output_digest(), d("output"));
        assert!(!active.grants_physical_authority());
        assert!(!active.hardware_certification_established());
    }

    #[test]
    fn executable_drift_blocks() {
        let policy = policy();
        let mut evidence = evidence();
        let mut checkpoint = evidence.checkpoints[1].clone();
        checkpoint.executable_digest = d("other-exe");
        checkpoint.signature_ed25519_hex = "00".repeat(64);
        evidence.checkpoints[1] = sign_checkpoint(checkpoint);
        let digest = evidence.checkpoints[1].canonical_digest().unwrap();
        let mut computation = evidence.computation.clone();
        computation.final_checkpoint_digest = digest;
        computation.signature_ed25519_hex = "00".repeat(64);
        evidence.computation = sign_computation(computation);
        let result = verify_continuous_verifier_execution(&policy, &evidence, 5_000);
        assert_eq!(result.report.disposition, RuntimeContinuityDisposition::Blocked);
        assert!(result.report.issues.iter().any(|issue| matches!(
            issue,
            RuntimeContinuityIssue::CheckpointStaticIdentityDrift { sequence: 2 }
        )));
    }

    #[test]
    fn checkpoint_gap_blocks() {
        let policy = policy();
        let mut evidence = evidence();
        let mut checkpoint = evidence.checkpoints[1].clone();
        checkpoint.observed_at_ms = 7_500;
        checkpoint.signature_ed25519_hex = "00".repeat(64);
        evidence.checkpoints[1] = sign_checkpoint(checkpoint);
        let digest = evidence.checkpoints[1].canonical_digest().unwrap();
        let mut computation = evidence.computation.clone();
        computation.final_checkpoint_digest = digest;
        computation.completed_at_ms = 7_400;
        computation.attested_at_ms = 7_600;
        computation.signature_ed25519_hex = "00".repeat(64);
        evidence.computation = sign_computation(computation);
        let result = verify_continuous_verifier_execution(&policy, &evidence, 8_000);
        assert_eq!(result.report.disposition, RuntimeContinuityDisposition::Blocked);
        assert!(result.report.issues.iter().any(|issue| matches!(
            issue,
            RuntimeContinuityIssue::CheckpointGapExceeded { sequence: 2, .. }
        )));
    }

    #[test]
    fn broken_checkpoint_link_is_invalid() {
        let policy = policy();
        let mut evidence = evidence();
        let mut checkpoint = evidence.checkpoints[1].clone();
        checkpoint.previous_checkpoint_digest = Some(d("wrong-parent"));
        checkpoint.signature_ed25519_hex = "00".repeat(64);
        evidence.checkpoints[1] = sign_checkpoint(checkpoint);
        let digest = evidence.checkpoints[1].canonical_digest().unwrap();
        let mut computation = evidence.computation.clone();
        computation.final_checkpoint_digest = digest;
        computation.signature_ed25519_hex = "00".repeat(64);
        evidence.computation = sign_computation(computation);
        let result = verify_continuous_verifier_execution(&policy, &evidence, 5_000);
        assert_eq!(result.report.disposition, RuntimeContinuityDisposition::Invalid);
    }

    #[test]
    fn computation_output_is_bound_into_capability_identity() {
        let policy = policy();
        let first = verify_continuous_verifier_execution(&policy, &evidence(), 5_000)
            .into_continuous()
            .unwrap();
        let mut changed = evidence();
        let mut computation = changed.computation.clone();
        computation.output_digest = d("different-output");
        computation.signature_ed25519_hex = "00".repeat(64);
        changed.computation = sign_computation(computation);
        let second = verify_continuous_verifier_execution(&policy, &changed, 5_000)
            .into_continuous()
            .unwrap();
        assert_ne!(first.continuity_digest(), second.continuity_digest());
    }

    impl RuntimeContinuityAssessment {
        fn active_or_continuous_for_test(&self) -> &ContinuousVerifierExecution {
            self.continuous().expect("qualified fixture")
        }
    }
}
