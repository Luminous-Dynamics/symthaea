// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

//! External cryptographic attestation adapter for observation-fabric receipts.
//!
//! This crate deliberately lives outside symthaea-core. The core defines the
//! receipt/envelope semantics and canonical payload; this adapter adds one concrete
//! cryptographic proof suite (detached Ed25519) without making cryptographic key
//! resolution, authorization, revocation, or substantive truth part of the core.
//!
//! The adapter is Symthaea-native. It is not a claim of W3C Data Integrity
//! conformance: W3C Data Integrity 1.1 is currently a Working Draft, and its
//! document-processing and cryptosuite rules are distinct from this envelope's
//! explicit canonical byte contract.

use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use symthaea_core::observation_fabric::{
    IndependenceVerificationReceipt, ReceiptAttestationEnvelope,
    ReceiptAttestationTemporalStatus,
};

pub const CRYPTOSUITE: &str = "symthaea-ed25519-detached-v1";
pub const VERIFIER_VERSION: &str = "symthaea-observation-attestation-report-v1";

pub const POLICY_VERSION: &str = "symthaea-observation-verification-policy-v1";
pub const VERIFIER_IMPLEMENTATION_ID: &str = "symthaea-observation-attestation-ed25519-v1";
pub const ENVIRONMENT_IDENTITY_VERSION: &str = "symthaea-verifier-environment-v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerificationPolicyInputs {
    pub policy_version: &'static str,
    pub cryptosuite: &'static str,
    pub expected_proof_purpose: Option<String>,
    pub expected_domain: Option<String>,
    pub expected_challenge_fingerprint: Option<String>,
    pub require_active_verification_method: bool,
}

impl VerificationPolicyInputs {
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        fn write_string(bytes: &mut Vec<u8>, value: &str) {
            bytes.extend_from_slice(&(value.len() as u64).to_be_bytes());
            bytes.extend_from_slice(value.as_bytes());
        }
        fn write_option(bytes: &mut Vec<u8>, value: Option<&str>) {
            match value {
                Some(value) => { bytes.push(1); write_string(bytes, value); }
                None => bytes.push(0),
            }
        }
        write_string(&mut bytes, self.policy_version);
        write_string(&mut bytes, self.cryptosuite);
        write_option(&mut bytes, self.expected_proof_purpose.as_deref());
        write_option(&mut bytes, self.expected_domain.as_deref());
        write_option(&mut bytes, self.expected_challenge_fingerprint.as_deref());
        bytes.push(self.require_active_verification_method as u8);
        bytes
    }

    pub fn fingerprint(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea:observation-verification-policy:v1\n");
        hasher.update(&self.canonical_bytes());
        hasher.finalize().to_hex().to_string()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VerifierEnvironmentIdentity {
    pub identity_version: &'static str,
    pub implementation_id: &'static str,
    pub build_fingerprint: String,
    pub runtime_profile: Option<String>,
}

impl VerifierEnvironmentIdentity {
    pub fn new(build_fingerprint: impl Into<String>) -> Self {
        Self {
            identity_version: ENVIRONMENT_IDENTITY_VERSION,
            implementation_id: VERIFIER_IMPLEMENTATION_ID,
            build_fingerprint: build_fingerprint.into(),
            runtime_profile: None,
        }
    }

    pub fn with_runtime_profile(mut self, profile: impl Into<String>) -> Self {
        self.runtime_profile = Some(profile.into());
        self
    }

    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        fn write_string(bytes: &mut Vec<u8>, value: &str) {
            bytes.extend_from_slice(&(value.len() as u64).to_be_bytes());
            bytes.extend_from_slice(value.as_bytes());
        }
        write_string(&mut bytes, self.identity_version);
        write_string(&mut bytes, self.implementation_id);
        write_string(&mut bytes, &self.build_fingerprint);
        match self.runtime_profile.as_deref() {
            Some(profile) => { bytes.push(1); write_string(&mut bytes, profile); }
            None => bytes.push(0),
        }
        bytes
    }

    pub fn fingerprint(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea:verifier-environment:v1\n");
        hasher.update(&self.canonical_bytes());
        hasher.finalize().to_hex().to_string()
    }
}

const REPORT_DOMAIN_SEPARATOR: &[u8] = b"symthaea:observation-attestation-report:v1\n";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReceiptAttestationVerificationOutcome {
    Verified,
    InvalidEnvelope,
    ReceiptCommitmentMismatch,
    NotYetValid,
    Expired,
    MissingProof,
    InvalidProofEncoding,
    CryptosuiteMismatch,
    VerificationMethodMismatch,
    ProofPurposeMismatch,
    DomainMismatch,
    ChallengeMismatch,
    InvalidSignature,
    VerificationMethodUnavailable,
    VerificationMethodRevoked,
    VerificationMethodExpired,
    ProofPurposeUnauthorized,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerificationStage {
    Passed,
    Failed(ReceiptAttestationVerificationOutcome),
    NotEvaluated,
}

/// Structured stage-by-stage verification evidence. This deliberately does not
/// collapse evidence into an aggregate trust score.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReceiptAttestationVerificationReport {
    pub outcome: ReceiptAttestationVerificationOutcome,
    pub verifier_version: &'static str,
    pub receipt_fingerprint: String,
    pub evaluated_at_unix_ns: i128,
    pub policy_inputs: VerificationPolicyInputs,
    pub policy_fingerprint: String,
    pub environment_identity: VerifierEnvironmentIdentity,
    pub environment_fingerprint: String,
    pub resolved_verification_method: Option<String>,
    pub structural_validation: VerificationStage,
    pub receipt_commitment: VerificationStage,
    pub temporal_validity: VerificationStage,
    pub cryptosuite: VerificationStage,
    pub verification_method: VerificationStage,
    pub lifecycle: VerificationStage,
    pub proof_purpose_authorization: VerificationStage,
    pub proof_policy: VerificationStage,
    pub cryptographic_proof: VerificationStage,
}

impl ReceiptAttestationVerificationReport {
    fn failed(
        outcome: ReceiptAttestationVerificationOutcome,
        stage: fn(ReceiptAttestationVerificationOutcome) -> VerificationStage,
        receipt_fingerprint: String,
        resolved_verification_method: Option<String>,
        evaluated_at_unix_ns: i128,
        policy_inputs: VerificationPolicyInputs,
        environment_identity: VerifierEnvironmentIdentity,
    ) -> Self {
        let failed = stage(outcome);
        let mut report = Self {
            outcome,
            verifier_version: VERIFIER_VERSION,
            receipt_fingerprint,
            evaluated_at_unix_ns,
            policy_fingerprint: policy_inputs.fingerprint(),
            environment_fingerprint: environment_identity.fingerprint(),
            policy_inputs,
            environment_identity,
            resolved_verification_method,
            structural_validation: VerificationStage::NotEvaluated,
            receipt_commitment: VerificationStage::NotEvaluated,
            temporal_validity: VerificationStage::NotEvaluated,
            cryptosuite: VerificationStage::NotEvaluated,
            verification_method: VerificationStage::NotEvaluated,
            lifecycle: VerificationStage::NotEvaluated,
            proof_purpose_authorization: VerificationStage::NotEvaluated,
            proof_policy: VerificationStage::NotEvaluated,
            cryptographic_proof: VerificationStage::NotEvaluated,
        };

        match outcome {
            ReceiptAttestationVerificationOutcome::InvalidEnvelope => {
                report.structural_validation = failed;
            }
            ReceiptAttestationVerificationOutcome::ReceiptCommitmentMismatch => {
                report.structural_validation = VerificationStage::Passed;
                report.receipt_commitment = failed;
            }
            ReceiptAttestationVerificationOutcome::NotYetValid
            | ReceiptAttestationVerificationOutcome::Expired => {
                report.structural_validation = VerificationStage::Passed;
                report.receipt_commitment = VerificationStage::Passed;
                report.temporal_validity = failed;
            }
            ReceiptAttestationVerificationOutcome::CryptosuiteMismatch => {
                report.structural_validation = VerificationStage::Passed;
                report.receipt_commitment = VerificationStage::Passed;
                report.temporal_validity = VerificationStage::Passed;
                report.cryptosuite = failed;
            }
            ReceiptAttestationVerificationOutcome::VerificationMethodMismatch
            | ReceiptAttestationVerificationOutcome::VerificationMethodUnavailable => {
                report.structural_validation = VerificationStage::Passed;
                report.receipt_commitment = VerificationStage::Passed;
                report.temporal_validity = VerificationStage::Passed;
                report.cryptosuite = VerificationStage::Passed;
                report.verification_method = failed;
            }
            ReceiptAttestationVerificationOutcome::VerificationMethodRevoked
            | ReceiptAttestationVerificationOutcome::VerificationMethodExpired => {
                report.structural_validation = VerificationStage::Passed;
                report.receipt_commitment = VerificationStage::Passed;
                report.temporal_validity = VerificationStage::Passed;
                report.cryptosuite = VerificationStage::Passed;
                report.verification_method = VerificationStage::Passed;
                report.lifecycle = failed;
            }
            ReceiptAttestationVerificationOutcome::ProofPurposeUnauthorized => {
                report.structural_validation = VerificationStage::Passed;
                report.receipt_commitment = VerificationStage::Passed;
                report.temporal_validity = VerificationStage::Passed;
                report.cryptosuite = VerificationStage::Passed;
                report.verification_method = VerificationStage::Passed;
                report.lifecycle = VerificationStage::Passed;
                report.proof_purpose_authorization = failed;
            }
            ReceiptAttestationVerificationOutcome::ProofPurposeMismatch
            | ReceiptAttestationVerificationOutcome::DomainMismatch
            | ReceiptAttestationVerificationOutcome::ChallengeMismatch => {
                report.structural_validation = VerificationStage::Passed;
                report.receipt_commitment = VerificationStage::Passed;
                report.temporal_validity = VerificationStage::Passed;
                report.cryptosuite = VerificationStage::Passed;
                report.verification_method = VerificationStage::Passed;
                report.lifecycle = VerificationStage::Passed;
                report.proof_purpose_authorization = VerificationStage::Passed;
                report.proof_policy = failed;
            }
            ReceiptAttestationVerificationOutcome::MissingProof
            | ReceiptAttestationVerificationOutcome::InvalidProofEncoding
            | ReceiptAttestationVerificationOutcome::InvalidSignature => {
                report.structural_validation = VerificationStage::Passed;
                report.receipt_commitment = VerificationStage::Passed;
                report.temporal_validity = VerificationStage::Passed;
                report.cryptosuite = VerificationStage::Passed;
                report.verification_method = VerificationStage::Passed;
                report.lifecycle = VerificationStage::Passed;
                report.proof_purpose_authorization = VerificationStage::Passed;
                report.proof_policy = VerificationStage::Passed;
                report.cryptographic_proof = failed;
            }
            ReceiptAttestationVerificationOutcome::Verified => {}
        }
        report
    }

    fn passed(
        receipt_fingerprint: String,
        resolved_verification_method: Option<String>,
    ) -> Self {
        Self {
            outcome: ReceiptAttestationVerificationOutcome::Verified,
            verifier_version: VERIFIER_VERSION,
            receipt_fingerprint,
            resolved_verification_method,
            structural_validation: VerificationStage::Passed,
            receipt_commitment: VerificationStage::Passed,
            temporal_validity: VerificationStage::Passed,
            cryptosuite: VerificationStage::Passed,
            verification_method: VerificationStage::Passed,
            lifecycle: VerificationStage::Passed,
            proof_purpose_authorization: VerificationStage::Passed,
            proof_policy: VerificationStage::Passed,
            cryptographic_proof: VerificationStage::Passed,
        }
    }
    
    pub fn canonical_bytes(&self) -> Vec<u8> {
        fn write_string(bytes: &mut Vec<u8>, value: &str) {
            bytes.extend_from_slice(&(value.len() as u64).to_le_bytes());
            bytes.extend_from_slice(value.as_bytes());
        }
        fn write_stage(bytes: &mut Vec<u8>, stage: VerificationStage) {
            match stage {
                VerificationStage::Passed => bytes.push(0),
                VerificationStage::Failed(outcome) => {
                    bytes.push(1);
                    bytes.push(outcome as u8);
                }
                VerificationStage::NotEvaluated => bytes.push(2),
            }
        }

        let mut bytes = Vec::new();
        bytes.extend_from_slice(REPORT_DOMAIN_SEPARATOR);
        write_string(&mut bytes, self.verifier_version);
        write_string(&mut bytes, &self.receipt_fingerprint);
        bytes.extend_from_slice(&self.evaluated_at_unix_ns.to_be_bytes());
        write_string(&mut bytes, &self.policy_fingerprint);
        write_string(&mut bytes, &self.environment_fingerprint);
        match &self.resolved_verification_method {
            Some(method) => {
                bytes.push(1);
                write_string(&mut bytes, method);
            }
            None => bytes.push(0),
        }
        bytes.push(self.outcome as u8);
        write_stage(&mut bytes, self.structural_validation);
        write_stage(&mut bytes, self.receipt_commitment);
        write_stage(&mut bytes, self.temporal_validity);
        write_stage(&mut bytes, self.cryptosuite);
        write_stage(&mut bytes, self.verification_method);
        write_stage(&mut bytes, self.lifecycle);
        write_stage(&mut bytes, self.proof_purpose_authorization);
        write_stage(&mut bytes, self.proof_policy);
        write_stage(&mut bytes, self.cryptographic_proof);
        bytes
    }

    pub fn fingerprint(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(b"symthaea:observation-attestation-report:v1\n");
        hasher.update(&self.canonical_bytes());
        hasher.finalize().to_hex().to_string()
    }

}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VerificationMethodStatus {
    Active,
    Revoked,
    Expired,
    Unknown,
}

#[derive(Debug, Clone)]
pub struct ResolvedVerificationMethod {
    pub verification_method: String,
    pub verifying_key: VerifyingKey,
    pub status: VerificationMethodStatus,
    pub allowed_proof_purposes: Vec<String>,
}

impl ResolvedVerificationMethod {
    pub fn is_authorized_for(&self, proof_purpose: &str) -> bool {
        self.allowed_proof_purposes
            .iter()
            .any(|purpose| purpose == proof_purpose)
    }
}

/// Application-supplied verification-method resolver.
///
/// Resolution, controller authorization, key lifecycle, and status are deliberately
/// injected rather than performed through network access in this crate.
pub trait VerificationMethodResolver {
    fn resolve(
        &self,
        verification_method: &str,
    ) -> Result<ResolvedVerificationMethod, VerificationMethodResolutionError>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum VerificationMethodResolutionError {
    #[error("verification method is unavailable")]
    Unavailable,
}

#[derive(Debug, Clone)]
pub struct InMemoryVerificationMethodResolver {
    methods: std::collections::BTreeMap<String, ResolvedVerificationMethod>,
}

impl InMemoryVerificationMethodResolver {
    pub fn new(methods: impl IntoIterator<Item = ResolvedVerificationMethod>) -> Self {
        Self {
            methods: methods
                .into_iter()
                .map(|method| (method.verification_method.clone(), method))
                .collect(),
        }
    }
}

impl VerificationMethodResolver for InMemoryVerificationMethodResolver {
    fn resolve(
        &self,
        verification_method: &str,
    ) -> Result<ResolvedVerificationMethod, VerificationMethodResolutionError> {
        self.methods
            .get(verification_method)
            .cloned()
            .ok_or(VerificationMethodResolutionError::Unavailable)
    }
}

#[derive(Debug, Clone)]
pub struct Ed25519ReceiptVerifier {
    verification_method: String,
    verifying_key: VerifyingKey,
    expected_proof_purpose: Option<String>,
    expected_domain: Option<String>,
    expected_challenge: Option<String>,
    now_unix_ns: i128,
    policy_inputs: VerificationPolicyInputs,
    environment_identity: VerifierEnvironmentIdentity,
}

impl Ed25519ReceiptVerifier {
    pub fn new(
        verification_method: impl Into<String>,
        verifying_key: VerifyingKey,
        now_unix_ns: i128,
    ) -> Self {
        Self {
            verification_method: verification_method.into(),
            verifying_key,
            expected_proof_purpose: None,
            expected_domain: None,
            expected_challenge: None,
            now_unix_ns,
            policy_inputs: VerificationPolicyInputs {
                policy_version: POLICY_VERSION,
                cryptosuite: CRYPTOSUITE,
                expected_proof_purpose: None,
                expected_domain: None,
                expected_challenge_fingerprint: None,
                require_active_verification_method: true,
            },
            environment_identity: VerifierEnvironmentIdentity::new("unspecified"),
        }
    }

    pub fn with_expected_proof_purpose(mut self, purpose: impl Into<String>) -> Self {
        self.expected_proof_purpose = Some(purpose.into());
        self.policy_inputs.expected_proof_purpose = self.expected_proof_purpose.clone();
        self
    }

    pub fn with_expected_domain(mut self, domain: impl Into<String>) -> Self {
        self.expected_domain = Some(domain.into());
        self.policy_inputs.expected_domain = self.expected_domain.clone();
        self
    }

    pub fn with_expected_challenge(mut self, challenge: impl Into<String>) -> Self {
        self.expected_challenge = Some(challenge.into());
        self.policy_inputs.expected_challenge_fingerprint = self.expected_challenge.as_deref().map(|v| blake3::hash(v.as_bytes()).to_hex().to_string());
        self
    }

    pub fn with_environment_identity(mut self, identity: VerifierEnvironmentIdentity) -> Self {
        self.environment_identity = identity;
        self
    }

    pub fn with_policy_version(mut self, version: &'static str) -> Self {
        self.policy_inputs.policy_version = version;
        self
    }

    /// Verify structure, receipt commitment, temporal status, policy bindings,
    /// and the detached Ed25519 proof. This does not perform issuer
    /// authorization, credential validation, revocation/status resolution,
    /// or substantive observation validation.
    pub fn verify(
        &self,
        envelope: &ReceiptAttestationEnvelope,
        receipt: &IndependenceVerificationReceipt,
    ) -> ReceiptAttestationVerificationOutcome {
        self.verify_report(envelope, receipt).outcome
    }

    pub fn verify_report(
        &self,
        envelope: &ReceiptAttestationEnvelope,
        receipt: &IndependenceVerificationReceipt,
    ) -> ReceiptAttestationVerificationReport {
        self.verify_with_resolved_key_report(
            envelope, receipt, &self.verification_method, &self.verifying_key,
        )
    }

    /// Verify using an application-controlled resolver.
    ///
    /// This adds verification-method resolution, lifecycle status, and proof-purpose
    /// authorization without allowing this crate to fetch or trust remote key
    /// material implicitly.
    pub fn verify_with_resolver<R: VerificationMethodResolver>(
        &self,
        envelope: &ReceiptAttestationEnvelope,
        receipt: &IndependenceVerificationReceipt,
        resolver: &R,
    ) -> ReceiptAttestationVerificationOutcome {
        self.verify_with_resolver_report(envelope, receipt, resolver).outcome
    }

    pub fn verify_with_resolver_report<R: VerificationMethodResolver>(
        &self,
        envelope: &ReceiptAttestationEnvelope,
        receipt: &IndependenceVerificationReceipt,
        resolver: &R,
    ) -> ReceiptAttestationVerificationReport {
        if envelope.validate().is_err() {
            return ReceiptAttestationVerificationReport::failed(
                ReceiptAttestationVerificationOutcome::InvalidEnvelope,
                VerificationStage::Failed,
                receipt.fingerprint(),
                None,,
                    self.now_unix_ns,
                    self.policy_inputs.clone(),
                    self.environment_identity.clone(),
            );
        }
        if !envelope.verify_against_receipt(receipt) {
            return ReceiptAttestationVerificationReport::failed(
                ReceiptAttestationVerificationOutcome::ReceiptCommitmentMismatch,
                VerificationStage::Failed,
                receipt.fingerprint(),
                None,,
                    self.now_unix_ns,
                    self.policy_inputs.clone(),
                    self.environment_identity.clone(),
            );
        }
        match envelope.temporal_status_at(self.now_unix_ns) {
            ReceiptAttestationTemporalStatus::NotYetValid => {
                return ReceiptAttestationVerificationReport::failed(
                    ReceiptAttestationVerificationOutcome::NotYetValid,
                    VerificationStage::Failed,
                    receipt.fingerprint(),
                    None,
                    self.now_unix_ns,
                    self.policy_inputs.clone(),
                    self.environment_identity.clone(),
                );
            }
            ReceiptAttestationTemporalStatus::Expired => {
                return ReceiptAttestationVerificationReport::failed(
                    ReceiptAttestationVerificationOutcome::Expired,
                    VerificationStage::Failed,
                    receipt.fingerprint(),
                    None,,
                    self.now_unix_ns,
                    self.policy_inputs.clone(),
                    self.environment_identity.clone(),
                );
            }
            ReceiptAttestationTemporalStatus::Valid => {}
        }
        if envelope.cryptosuite.as_deref() != Some(CRYPTOSUITE) {
            return ReceiptAttestationVerificationReport::failed(
                ReceiptAttestationVerificationOutcome::CryptosuiteMismatch,
                VerificationStage::Failed,
                receipt.fingerprint(),
                None,,
                    self.now_unix_ns,
                    self.policy_inputs.clone(),
                    self.environment_identity.clone(),
            );
        }

        let Some(method) = envelope.verification_method.as_deref() else {
            return ReceiptAttestationVerificationReport::failed(
                ReceiptAttestationVerificationOutcome::VerificationMethodUnavailable,
                VerificationStage::Failed,
                receipt.fingerprint(),
                None,,
                    self.now_unix_ns,
                    self.policy_inputs.clone(),
                    self.environment_identity.clone(),
            );
        };
        let resolved = match resolver.resolve(method) {
            Ok(resolved) => resolved,
            Err(_) => {
                return ReceiptAttestationVerificationReport::failed(
                    ReceiptAttestationVerificationOutcome::VerificationMethodUnavailable,
                    VerificationStage::Failed,
                    receipt.fingerprint(),
                    Some(method.to_string()),
                    self.now_unix_ns,
                    self.policy_inputs.clone(),
                    self.environment_identity.clone(),
                );
            }
        };
        if resolved.verification_method != method {
            return ReceiptAttestationVerificationReport::failed(
                ReceiptAttestationVerificationOutcome::VerificationMethodUnavailable,
                VerificationStage::Failed,
                receipt.fingerprint(),
                Some(method.to_string()),,
                    self.now_unix_ns,
                    self.policy_inputs.clone(),
                    self.environment_identity.clone(),
            );
        }
        match resolved.status {
            VerificationMethodStatus::Active => {}
            VerificationMethodStatus::Revoked => {
                return ReceiptAttestationVerificationReport::failed(
                    ReceiptAttestationVerificationOutcome::VerificationMethodRevoked,
                    VerificationStage::Failed,
                    receipt.fingerprint(),
                    Some(method.to_string()),,
                    self.now_unix_ns,
                    self.policy_inputs.clone(),
                    self.environment_identity.clone(),
                );
            }
            VerificationMethodStatus::Expired => {
                return ReceiptAttestationVerificationReport::failed(
                    ReceiptAttestationVerificationOutcome::VerificationMethodExpired,
                    VerificationStage::Failed,
                    receipt.fingerprint(),
                    Some(method.to_string()),,
                    self.now_unix_ns,
                    self.policy_inputs.clone(),
                    self.environment_identity.clone(),
                );
            }
            VerificationMethodStatus::Unknown => {
                return ReceiptAttestationVerificationReport::failed(
                    ReceiptAttestationVerificationOutcome::VerificationMethodUnavailable,
                    VerificationStage::Failed,
                    receipt.fingerprint(),
                    Some(method.to_string()),,
                    self.now_unix_ns,
                    self.policy_inputs.clone(),
                    self.environment_identity.clone(),
                );
            }
        }
        if !resolved.is_authorized_for(&envelope.proof_purpose) {
            return ReceiptAttestationVerificationReport::failed(
                ReceiptAttestationVerificationOutcome::ProofPurposeUnauthorized,
                VerificationStage::Failed,
                receipt.fingerprint(),
                Some(method.to_string()),,
                    self.now_unix_ns,
                    self.policy_inputs.clone(),
                    self.environment_identity.clone(),
            );
        }
        self.verify_with_resolved_key_report(envelope, receipt, method, &resolved.verifying_key)
    }

    fn verify_with_resolved_key(
        &self,
        envelope: &ReceiptAttestationEnvelope,
        receipt: &IndependenceVerificationReceipt,
        verification_method: &str,
        verifying_key: &VerifyingKey,
    ) -> ReceiptAttestationVerificationOutcome {
        self.verify_with_resolved_key_report(
            envelope, receipt, verification_method, verifying_key,
        ).outcome
    }

    fn verify_with_resolved_key_report(
        &self,
        envelope: &ReceiptAttestationEnvelope,
        receipt: &IndependenceVerificationReceipt,
        verification_method: &str,
        verifying_key: &VerifyingKey,
    ) -> ReceiptAttestationVerificationReport {
        let method = Some(verification_method.to_string());
        let fingerprint = receipt.fingerprint();

        if envelope.validate().is_err() {
            return ReceiptAttestationVerificationReport::failed(
                ReceiptAttestationVerificationOutcome::InvalidEnvelope,
                VerificationStage::Failed,
                fingerprint,
                method,
                    self.now_unix_ns,
                    self.policy_inputs.clone(),
                    self.environment_identity.clone(),
            );
        }
        if !envelope.verify_against_receipt(receipt) {
            return ReceiptAttestationVerificationReport::failed(
                ReceiptAttestationVerificationOutcome::ReceiptCommitmentMismatch,
                VerificationStage::Failed,
                fingerprint,
                method,
                    self.now_unix_ns,
                    self.policy_inputs.clone(),
                    self.environment_identity.clone(),
            );
        }

        match envelope.temporal_status_at(self.now_unix_ns) {
            ReceiptAttestationTemporalStatus::NotYetValid => {
                return ReceiptAttestationVerificationReport::failed(
                    ReceiptAttestationVerificationOutcome::NotYetValid,
                    VerificationStage::Failed,
                    fingerprint,
                    method,
                    self.now_unix_ns,
                    self.policy_inputs.clone(),
                    self.environment_identity.clone(),
                );
            }
            ReceiptAttestationTemporalStatus::Expired => {
                return ReceiptAttestationVerificationReport::failed(
                    ReceiptAttestationVerificationOutcome::Expired,
                    VerificationStage::Failed,
                    fingerprint,
                    method,
                    self.now_unix_ns,
                    self.policy_inputs.clone(),
                    self.environment_identity.clone(),
                );
            }
            ReceiptAttestationTemporalStatus::Valid => {}
        }

        if envelope.cryptosuite.as_deref() != Some(CRYPTOSUITE) {
            return ReceiptAttestationVerificationReport::failed(
                ReceiptAttestationVerificationOutcome::CryptosuiteMismatch,
                VerificationStage::Failed,
                fingerprint,
                method,
                    self.now_unix_ns,
                    self.policy_inputs.clone(),
                    self.environment_identity.clone(),
            );
        }
        if envelope.verification_method.as_deref() != Some(verification_method) {
            return ReceiptAttestationVerificationReport::failed(
                ReceiptAttestationVerificationOutcome::VerificationMethodMismatch,
                VerificationStage::Failed,
                fingerprint,
                method,
                    self.now_unix_ns,
                    self.policy_inputs.clone(),
                    self.environment_identity.clone(),
            );
        }
        if let Some(expected) = &self.expected_proof_purpose {
            if envelope.proof_purpose != *expected {
                return ReceiptAttestationVerificationReport::failed(
                    ReceiptAttestationVerificationOutcome::ProofPurposeMismatch,
                    VerificationStage::Failed,
                    fingerprint,
                    method,
                    self.now_unix_ns,
                    self.policy_inputs.clone(),
                    self.environment_identity.clone(),
                );
            }
        }
        if let Some(expected) = &self.expected_domain {
            if envelope.domain.as_deref() != Some(expected.as_str()) {
                return ReceiptAttestationVerificationReport::failed(
                    ReceiptAttestationVerificationOutcome::DomainMismatch,
                    VerificationStage::Failed,
                    fingerprint,
                    method,
                    self.now_unix_ns,
                    self.policy_inputs.clone(),
                    self.environment_identity.clone(),
                );
            }
        }
        if let Some(expected) = &self.expected_challenge {
            if envelope.challenge.as_deref() != Some(expected.as_str()) {
                return ReceiptAttestationVerificationReport::failed(
                    ReceiptAttestationVerificationOutcome::ChallengeMismatch,
                    VerificationStage::Failed,
                    fingerprint,
                    method,
                    self.now_unix_ns,
                    self.policy_inputs.clone(),
                    self.environment_identity.clone(),
                );
            }
        }

        let Some(proof) = envelope.proof.as_deref() else {
            return ReceiptAttestationVerificationReport::failed(
                ReceiptAttestationVerificationOutcome::MissingProof,
                VerificationStage::Failed,
                fingerprint,
                method,
                    self.now_unix_ns,
                    self.policy_inputs.clone(),
                    self.environment_identity.clone(),
            );
        };
        let Ok(proof_bytes) = <[u8; 64]>::try_from(proof) else {
            return ReceiptAttestationVerificationReport::failed(
                ReceiptAttestationVerificationOutcome::InvalidProofEncoding,
                VerificationStage::Failed,
                fingerprint,
                method,
                    self.now_unix_ns,
                    self.policy_inputs.clone(),
                    self.environment_identity.clone(),
            );
        };
        let signature = Signature::from_bytes(&proof_bytes);

        match verifying_key.verify(&envelope.canonical_payload_bytes(), &signature) {
            Ok(()) => ReceiptAttestationVerificationReport::passed(
                fingerprint,
                method,
                self.now_unix_ns,
                self.policy_inputs.clone(),
                self.environment_identity.clone(),
            ),
            Err(_) => ReceiptAttestationVerificationReport::failed(
                ReceiptAttestationVerificationOutcome::InvalidSignature,
                VerificationStage::Failed,
                fingerprint,
                method,
            ),
        }
    }
}

/// Attach a detached Ed25519 proof. Key publication, authorization, rotation,
/// and revocation remain outside this adapter.
pub fn sign_envelope(
    envelope: &mut ReceiptAttestationEnvelope,
    signing_key: &SigningKey,
    verification_method: impl Into<String>,
) -> Result<(), SignEnvelopeError> {
    envelope.cryptosuite = Some(CRYPTOSUITE.to_string());
    envelope.verification_method = Some(verification_method.into());
    envelope.validate().map_err(|_| SignEnvelopeError::InvalidEnvelope)?;
    let signature = signing_key.sign(&envelope.canonical_payload_bytes());
    envelope.proof = Some(signature.to_bytes().to_vec());
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum SignEnvelopeError {
    #[error("receipt attestation envelope is structurally invalid")]
    InvalidEnvelope,
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_core::observation_fabric::{
        AssetRef, DisclosurePolicy, Observation, ObservationGraph, ObservationModality,
        ObservationProvenance, ObservationQuality, ObservationTime, ProvenanceCoverage,
        ProvenanceVerification, SensorIdentity,
    };

    fn receipt() -> IndependenceVerificationReceipt {
        let base = Observation {
            id: "obs-a".into(),
            modality: ObservationModality::Optical,
            time: ObservationTime { observed_at_unix_ns: 100, time_uncertainty_ns: 1 },
            location: None,
            feature_of_interest_id: None,
            quality: ObservationQuality {
                confidence: 0.9,
                measurement_uncertainty: None,
                calibrated: true,
            },
            provenance: ObservationProvenance {
                source: SensorIdentity::new("sensor-a"),
                verification: ProvenanceVerification::Unverified,
                coverage: ProvenanceCoverage::Complete,
                attestation_id: None,
                acquired_by: None,
                parent_observation_ids: Vec::new(),
                processing_fingerprint: None,
                processing_activity: None,
            },
            asset: Some(AssetRef::blake3(b"a")),
            disclosure: DisclosurePolicy::restricted(),
        };
        let mut other = base.clone();
        other.id = "obs-b".into();
        other.provenance.source = SensorIdentity::new("sensor-b");
        let graph = ObservationGraph { observations: vec![base, other], relations: vec![] };
        let assessment = graph.assess_independence_detailed("obs-a", "obs-b").expect("assessment");
        IndependenceVerificationReceipt::from_assessment(&assessment)
    }

    fn envelope_and_key() -> (
        ReceiptAttestationEnvelope,
        SigningKey,
        IndependenceVerificationReceipt,
    ) {
        let receipt = receipt();
        let signing_key = SigningKey::from_bytes(&[7u8; 32]);
        let mut envelope = ReceiptAttestationEnvelope::from_receipt(
            &receipt, "attester-a", "observation-independence", 100,
        );
        envelope.expires_at_unix_ns = Some(200);
        sign_envelope(&mut envelope, &signing_key, "did:example:attester-a#key-1")
            .expect("sign");
        (envelope, signing_key, receipt)
    }

    #[test]
    fn signs_and_verifies_detached_receipt() {
        let (envelope, signing_key, receipt) = envelope_and_key();
        let verifier = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            150,
        );
        assert_eq!(
            verifier.verify(&envelope, &receipt),
            ReceiptAttestationVerificationOutcome::Verified
        );
    }

    #[test]
    fn payload_mutation_invalidates_signature() {
        let (mut envelope, signing_key, receipt) = envelope_and_key();
        let verifier = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            150,
        );
        assert_eq!(
            verifier.verify(&envelope, &receipt),
            ReceiptAttestationVerificationOutcome::Verified
        );
        envelope.challenge = Some("challenge".into());
        assert_eq!(
            verifier.verify(&envelope, &receipt),
            ReceiptAttestationVerificationOutcome::InvalidSignature
        );
    }

    #[test]
    fn temporal_and_key_policy_failures_are_distinct() {
        let (mut envelope, signing_key, receipt) = envelope_and_key();
        let verifier = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            200,
        );
        assert_eq!(
            verifier.verify(&envelope, &receipt),
            ReceiptAttestationVerificationOutcome::Expired
        );

        envelope.expires_at_unix_ns = Some(300);
        let verifier = Ed25519ReceiptVerifier::new(
            "did:example:wrong#key-1",
            signing_key.verifying_key(),
            150,
        );
        assert_eq!(
            verifier.verify(&envelope, &receipt),
            ReceiptAttestationVerificationOutcome::VerificationMethodMismatch
        );
    }

    #[test]
    fn resolver_separates_key_lifecycle_and_authorization() {
        let (envelope, signing_key, receipt) = envelope_and_key();
        let method = ResolvedVerificationMethod {
            verification_method: "did:example:attester-a#key-1".into(),
            verifying_key: signing_key.verifying_key(),
            status: VerificationMethodStatus::Active,
            allowed_proof_purposes: vec!["observation-independence".into()],
        };
        let resolver = InMemoryVerificationMethodResolver::new([method]);
        let verifier = Ed25519ReceiptVerifier::new(
            "ignored-by-resolver",
            signing_key.verifying_key(),
            150,
        );
        assert_eq!(
            verifier.verify_with_resolver(&envelope, &receipt, &resolver),
            ReceiptAttestationVerificationOutcome::Verified
        );

        let revoked = ResolvedVerificationMethod {
            verification_method: "did:example:attester-a#key-1".into(),
            verifying_key: signing_key.verifying_key(),
            status: VerificationMethodStatus::Revoked,
            allowed_proof_purposes: vec!["observation-independence".into()],
        };
        let resolver = InMemoryVerificationMethodResolver::new([revoked]);
        assert_eq!(
            verifier.verify_with_resolver(&envelope, &receipt, &resolver),
            ReceiptAttestationVerificationOutcome::VerificationMethodRevoked
        );

        let unauthorized = ResolvedVerificationMethod {
            verification_method: "did:example:attester-a#key-1".into(),
            verifying_key: signing_key.verifying_key(),
            status: VerificationMethodStatus::Active,
            allowed_proof_purposes: vec!["authentication".into()],
        };
        let resolver = InMemoryVerificationMethodResolver::new([unauthorized]);
        assert_eq!(
            verifier.verify_with_resolver(&envelope, &receipt, &resolver),
            ReceiptAttestationVerificationOutcome::ProofPurposeUnauthorized
        );
    }

    #[test]
    fn resolver_mode_rejects_invalid_envelope_before_resolution() {
        let (mut envelope, signing_key, receipt) = envelope_and_key();
        envelope.attester_id.clear();
        let resolver = InMemoryVerificationMethodResolver::new([]);
        let verifier = Ed25519ReceiptVerifier::new(
            "ignored-by-resolver",
            signing_key.verifying_key(),
            150,
        );

        assert_eq!(
            verifier.verify_with_resolver(&envelope, &receipt, &resolver),
            ReceiptAttestationVerificationOutcome::InvalidEnvelope
        );
    }

    #[test]
    fn resolver_uses_resolved_key_not_embedded_verifier_key() {
        let (envelope, signing_key, receipt) = envelope_and_key();
        let unrelated_key = SigningKey::from_bytes(&[9u8; 32]);
        let method = ResolvedVerificationMethod {
            verification_method: "did:example:attester-a#key-1".into(),
            verifying_key: signing_key.verifying_key(),
            status: VerificationMethodStatus::Active,
            allowed_proof_purposes: vec!["observation-independence".into()],
        };
        let resolver = InMemoryVerificationMethodResolver::new([method]);
        let verifier = Ed25519ReceiptVerifier::new(
            "ignored-by-resolver",
            unrelated_key.verifying_key(),
            150,
        );

        assert_eq!(
            verifier.verify_with_resolver(&envelope, &receipt, &resolver),
            ReceiptAttestationVerificationOutcome::Verified
        );
    }

    #[test]
    fn proof_purpose_domain_and_challenge_are_policy_checks() {
        let (mut envelope, signing_key, receipt) = envelope_and_key();
        envelope.domain = Some("example.org".into());
        envelope.challenge = Some("c1".into());
        sign_envelope(&mut envelope, &signing_key, "did:example:attester-a#key-1")
            .expect("resign");
        let verifier = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            150,
        )
        .with_expected_proof_purpose("observation-independence")
        .with_expected_domain("example.org")
        .with_expected_challenge("c1");
        assert_eq!(
            verifier.verify(&envelope, &receipt),
            ReceiptAttestationVerificationOutcome::Verified
        );
    }

    #[test]
    fn verification_report_preserves_stage_boundaries() {
        let (envelope, signing_key, receipt) = envelope_and_key();
        let verifier = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            150,
        );
        let report = verifier.verify_report(&envelope, &receipt);
        assert_eq!(report.outcome, ReceiptAttestationVerificationOutcome::Verified);
        assert_eq!(report.structural_validation, VerificationStage::Passed);
        assert_eq!(report.receipt_commitment, VerificationStage::Passed);
        assert_eq!(report.cryptographic_proof, VerificationStage::Passed);
    }

    #[test]
    fn verification_report_marks_unexecuted_stages() {
        let (mut envelope, signing_key, receipt) = envelope_and_key();
        envelope.attester_id.clear();
        let verifier = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            150,
        );
        let report = verifier.verify_report(&envelope, &receipt);
        assert_eq!(
            report.outcome,
            ReceiptAttestationVerificationOutcome::InvalidEnvelope
        );
        assert_eq!(
            report.structural_validation,
            VerificationStage::Failed(ReceiptAttestationVerificationOutcome::InvalidEnvelope)
        );
        assert_eq!(report.receipt_commitment, VerificationStage::NotEvaluated);
        assert_eq!(report.cryptographic_proof, VerificationStage::NotEvaluated);
    }

    #[test]
    fn resolver_report_records_lifecycle_failure() {
        let (envelope, signing_key, receipt) = envelope_and_key();
        let method = ResolvedVerificationMethod {
            verification_method: "did:example:attester-a#key-1".into(),
            verifying_key: signing_key.verifying_key(),
            status: VerificationMethodStatus::Revoked,
            allowed_proof_purposes: vec!["observation-independence".into()],
        };
        let resolver = InMemoryVerificationMethodResolver::new([method]);
        let verifier = Ed25519ReceiptVerifier::new(
            "ignored-by-resolver",
            signing_key.verifying_key(),
            150,
        );
        let report = verifier.verify_with_resolver_report(&envelope, &receipt, &resolver);
        assert_eq!(
            report.outcome,
            ReceiptAttestationVerificationOutcome::VerificationMethodRevoked
        );
        assert_eq!(report.verification_method, VerificationStage::Passed);
        assert_eq!(
            report.lifecycle,
            VerificationStage::Failed(
                ReceiptAttestationVerificationOutcome::VerificationMethodRevoked
            )
        );
        assert_eq!(
            report.proof_policy,
            VerificationStage::NotEvaluated
        );
    }


    #[test]
    fn verification_report_binds_policy_environment_and_evaluation_time() {
        let (envelope, signing_key, receipt) = envelope_and_key();
        let verifier = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            150,
        )
        .with_expected_proof_purpose("observation-independence")
        .with_expected_domain("mycelix")
        .with_environment_identity(
            VerifierEnvironmentIdentity::new("build-sha-abc")
                .with_runtime_profile("portable"),
        );
        let report = verifier.verify_report(&envelope, &receipt);
        assert_eq!(report.evaluated_at_unix_ns, 150);
        assert_eq!(report.policy_inputs.expected_proof_purpose.as_deref(), Some("observation-independence"));
        assert_eq!(report.policy_inputs.expected_domain.as_deref(), Some("mycelix"));
        assert_eq!(report.environment_identity.build_fingerprint, "build-sha-abc");
        assert_eq!(report.policy_fingerprint, report.policy_inputs.fingerprint());
        assert_eq!(report.environment_fingerprint, report.environment_identity.fingerprint());
    }

    #[test]
    fn verification_report_fingerprint_changes_when_policy_or_environment_changes() {
        let (envelope, signing_key, receipt) = envelope_and_key();
        let base = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            150,
        );
        let policy = base.clone().with_expected_domain("mycelix");
        let environment = base.clone().with_environment_identity(
            VerifierEnvironmentIdentity::new("build-sha-abc")
        );
        assert_ne!(
            base.verify_report(&envelope, &receipt).fingerprint(),
            policy.verify_report(&envelope, &receipt).fingerprint()
        );
        assert_ne!(
            base.verify_report(&envelope, &receipt).fingerprint(),
            environment.verify_report(&envelope, &receipt).fingerprint()
        );
    }

    #[test]
    fn verification_report_fingerprint_is_deterministic_and_binds_receipt() {
        let (envelope, signing_key, receipt) = envelope_and_key();
        let verifier = Ed25519ReceiptVerifier::new(
            "did:example:attester-a#key-1",
            signing_key.verifying_key(),
            150,
        );
        let first = verifier.verify_report(&envelope, &receipt);
        let second = verifier.verify_report(&envelope, &receipt);
        assert_eq!(first.fingerprint(), second.fingerprint());
        assert_eq!(first.receipt_fingerprint, receipt.fingerprint());

        let other_receipt = {
            let mut r = receipt.clone();
            r.source_observation_id = "different-source".into();
            r
        };
        assert_ne!(first.receipt_fingerprint, other_receipt.fingerprint());
    }


}
