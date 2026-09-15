// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! In-process TPM2 ECDSA P-256/SHA-256 quote signature assurance.
//!
//! This bridge deliberately reuses the existing raw-quote semantic verifier.
//! It replaces only that verifier's external `tpm2_checkquote` execution with
//! an in-process ECDSA verification backend and binds the operation to the
//! exact running host executable backing file observed through `/proc/self/exe`.

#![deny(unsafe_code)]

use aws_lc_rs::signature::{UnparsedPublicKey, ECDSA_P256_SHA256_FIXED};
use base64::Engine;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use symthaea_assurance_tpm2_attestation_possession::{
    AttestationChallenge, AttestationKeyBinding,
};
use symthaea_assurance_tpm2_checkquote_adapter::{
    verify_tpm2_quote, CheckquoteExecutionRequest, CheckquoteExecutor, RawTpm2QuoteBundle,
    ToolExecution, Tpm2CheckquoteDisposition, Tpm2CheckquotePolicy, Tpm2QuoteQualification,
};

pub const IN_PROCESS_TPM2_ECDSA_POLICY_SCHEMA_V1: &str =
    "symthaea.assurance.tpm2-in-process-ecdsa-policy.v1";
pub const IN_PROCESS_TPM2_ECDSA_REPORT_SCHEMA_V1: &str =
    "symthaea.assurance.tpm2-in-process-ecdsa-report.v1";
pub const IN_PROCESS_BACKEND_ID_V1: &str =
    "aws-lc-rs-1.17.0-ecdsa-p256-sha256-fixed-v1";
pub const SUPPORTED_AK_SIGNING_SCHEME_V1: &str = "ecdsa-sha256";

const POLICY_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.tpm2-in-process-ecdsa-policy.digest.v1\0";
const HOST_IDENTITY_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.tpm2-in-process-host-identity.digest.v1\0";
const REPORT_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.tpm2-in-process-ecdsa-report.digest.v1\0";
const QUALIFICATION_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.tpm2-in-process-ecdsa-qualification.digest.v1\0";
const MAX_TEXT_BYTES: usize = 1024;
const MAX_EVIDENCE_REFS: usize = 128;
const MAX_HOST_EXECUTABLE_BYTES: u64 = 512 * 1024 * 1024;
const TPM2_ALG_SHA256: u16 = 0x000b;
const TPM2_ALG_ECDSA: u16 = 0x0018;
const NIX_BASE32: &[u8] = b"0123456789abcdfghijklmnpqrsvwxyz";
const EC_PUBLIC_KEY_OID: &[u8] = &[0x2a, 0x86, 0x48, 0xce, 0x3d, 0x02, 0x01];
const PRIME256V1_OID: &[u8] = &[0x2a, 0x86, 0x48, 0xce, 0x3d, 0x03, 0x01, 0x07];

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InProcessTpm2EcdsaPolicy {
    pub schema_version: String,
    pub policy_id: String,
    pub expected_parent_policy_digest: String,
    pub expected_host_executable_path: String,
    pub expected_host_executable_blake3: String,
    pub expected_backend_id: String,
    pub max_host_executable_bytes: u64,
    pub evidence_refs: Vec<String>,
}

impl InProcessTpm2EcdsaPolicy {
    pub fn validate(&self) -> bool {
        self.schema_version == IN_PROCESS_TPM2_ECDSA_POLICY_SCHEMA_V1
            && canonical_text(&self.policy_id)
            && valid_blake3_digest(&self.expected_parent_policy_digest)
            && direct_nix_store_root(&self.expected_host_executable_path).is_some()
            && valid_blake3_digest(&self.expected_host_executable_blake3)
            && self.expected_backend_id == IN_PROCESS_BACKEND_ID_V1
            && (1..=MAX_HOST_EXECUTABLE_BYTES).contains(&self.max_host_executable_bytes)
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
            self.expected_parent_policy_digest.as_str(),
            self.expected_host_executable_path.as_str(),
            self.expected_host_executable_blake3.as_str(),
            self.expected_backend_id.as_str(),
        ] {
            push_field(&mut hasher, field);
        }
        hasher.update(&self.max_host_executable_bytes.to_le_bytes());
        push_sorted_refs(&mut hasher, &self.evidence_refs);
        Some(format!("blake3:{}", hasher.finalize().to_hex()))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RunningHostExecutableIdentity {
    pub canonical_path: String,
    pub nix_store_root: String,
    pub executable_blake3: String,
    pub device: u64,
    pub inode: u64,
    pub file_size: u64,
    pub mode: u32,
    pub uid: u32,
    pub gid: u32,
    pub mtime_seconds: i64,
    pub mtime_nanoseconds: i64,
}

impl RunningHostExecutableIdentity {
    pub fn canonical_digest(&self) -> Option<String> {
        if direct_nix_store_root(&self.canonical_path).as_deref()
            != Some(self.nix_store_root.as_str())
            || !valid_blake3_digest(&self.executable_blake3)
            || self.file_size == 0
            || self.uid != 0
            || self.mode & 0o222 != 0
            || self.mode & 0o111 == 0
        {
            return None;
        }
        let mut hasher = blake3::Hasher::new();
        hasher.update(HOST_IDENTITY_DIGEST_DOMAIN);
        for field in [
            self.canonical_path.as_str(),
            self.nix_store_root.as_str(),
            self.executable_blake3.as_str(),
        ] {
            push_field(&mut hasher, field);
        }
        for value in [self.device, self.inode, self.file_size] {
            hasher.update(&value.to_le_bytes());
        }
        for value in [self.mode, self.uid, self.gid] {
            hasher.update(&value.to_le_bytes());
        }
        hasher.update(&self.mtime_seconds.to_le_bytes());
        hasher.update(&self.mtime_nanoseconds.to_le_bytes());
        Some(format!("blake3:{}", hasher.finalize().to_hex()))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum InProcessTpm2EcdsaDisposition {
    Invalid,
    Blocked,
    Qualified,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum InProcessTpm2EcdsaIssue {
    InvalidPolicy,
    InvalidParentPolicy,
    ParentPolicyDigestMismatch,
    ParentVerifierPathMismatch,
    ParentVerifierDigestMismatch,
    UnsupportedAkSigningScheme,
    UnsupportedParentHashAlgorithm,
    UnsupportedAkPublicEncoding,
    HostIdentityUnavailable(String),
    HostPathMismatch,
    HostExecutableDigestMismatch,
    HostMetadataMismatch,
    ParentVerificationBlocked,
    ParentVerificationInvalid,
}

impl InProcessTpm2EcdsaIssue {
    fn is_invalid(&self) -> bool {
        matches!(
            self,
            Self::InvalidPolicy | Self::InvalidParentPolicy | Self::ParentVerificationInvalid
        )
    }

    fn code(&self) -> String {
        match self {
            Self::InvalidPolicy => "invalid-policy".into(),
            Self::InvalidParentPolicy => "invalid-parent-policy".into(),
            Self::ParentPolicyDigestMismatch => "parent-policy-digest-mismatch".into(),
            Self::ParentVerifierPathMismatch => "parent-verifier-path-mismatch".into(),
            Self::ParentVerifierDigestMismatch => "parent-verifier-digest-mismatch".into(),
            Self::UnsupportedAkSigningScheme => "unsupported-ak-signing-scheme".into(),
            Self::UnsupportedParentHashAlgorithm => "unsupported-parent-hash-algorithm".into(),
            Self::UnsupportedAkPublicEncoding => "unsupported-ak-public-encoding".into(),
            Self::HostIdentityUnavailable(reason) => format!("host-identity-unavailable:{reason}"),
            Self::HostPathMismatch => "host-path-mismatch".into(),
            Self::HostExecutableDigestMismatch => "host-executable-digest-mismatch".into(),
            Self::HostMetadataMismatch => "host-metadata-mismatch".into(),
            Self::ParentVerificationBlocked => "parent-verification-blocked".into(),
            Self::ParentVerificationInvalid => "parent-verification-invalid".into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct InProcessTpm2EcdsaReport {
    pub schema_version: String,
    pub policy_id: String,
    pub policy_digest: Option<String>,
    pub parent_policy_digest: Option<String>,
    pub parent_report_digest: Option<String>,
    pub parent_qualification_digest: Option<String>,
    pub host_identity_digest: Option<String>,
    pub host_executable_path: String,
    pub host_nix_store_root: String,
    pub host_executable_blake3: String,
    pub backend_id: String,
    pub disposition: InProcessTpm2EcdsaDisposition,
    pub issues: Vec<InProcessTpm2EcdsaIssue>,
}

impl InProcessTpm2EcdsaReport {
    pub fn canonical_digest(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(REPORT_DIGEST_DOMAIN);
        for field in [
            self.schema_version.as_str(),
            self.policy_id.as_str(),
            self.policy_digest.as_deref().unwrap_or("-"),
            self.parent_policy_digest.as_deref().unwrap_or("-"),
            self.parent_report_digest.as_deref().unwrap_or("-"),
            self.parent_qualification_digest.as_deref().unwrap_or("-"),
            self.host_identity_digest.as_deref().unwrap_or("-"),
            self.host_executable_path.as_str(),
            self.host_nix_store_root.as_str(),
            self.host_executable_blake3.as_str(),
            self.backend_id.as_str(),
        ] {
            push_field(&mut hasher, field);
        }
        push_field(
            &mut hasher,
            match self.disposition {
                InProcessTpm2EcdsaDisposition::Invalid => "invalid",
                InProcessTpm2EcdsaDisposition::Blocked => "blocked",
                InProcessTpm2EcdsaDisposition::Qualified => "qualified",
            },
        );
        for issue in &self.issues {
            push_field(&mut hasher, &issue.code());
        }
        format!("blake3:{}", hasher.finalize().to_hex())
    }

    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedInProcessTpm2Quote {
    qualification_digest: String,
    report_digest: String,
    policy_digest: String,
    parent_policy_digest: String,
    parent_qualification_digest: String,
    challenge_digest: String,
    ak_binding_digest: String,
    quote_artifact_digest: String,
    verification_receipt_digest: String,
    pcr_selection_digest: String,
    host_identity_digest: String,
    host_executable_path: String,
    host_nix_store_root: String,
    host_executable_blake3: String,
    host_device: u64,
    host_inode: u64,
    host_file_size: u64,
    backend_id: String,
    collected_at_ms: u64,
    verified_at_ms: u64,
}

impl VerifiedInProcessTpm2Quote {
    pub fn qualification_digest(&self) -> &str { &self.qualification_digest }
    pub fn report_digest(&self) -> &str { &self.report_digest }
    pub fn policy_digest(&self) -> &str { &self.policy_digest }
    pub fn parent_policy_digest(&self) -> &str { &self.parent_policy_digest }
    pub fn parent_qualification_digest(&self) -> &str { &self.parent_qualification_digest }
    pub fn challenge_digest(&self) -> &str { &self.challenge_digest }
    pub fn ak_binding_digest(&self) -> &str { &self.ak_binding_digest }
    pub fn quote_artifact_digest(&self) -> &str { &self.quote_artifact_digest }
    pub fn verification_receipt_digest(&self) -> &str { &self.verification_receipt_digest }
    pub fn pcr_selection_digest(&self) -> &str { &self.pcr_selection_digest }
    pub fn host_identity_digest(&self) -> &str { &self.host_identity_digest }
    pub fn host_executable_path(&self) -> &str { &self.host_executable_path }
    pub fn host_nix_store_root(&self) -> &str { &self.host_nix_store_root }
    pub fn host_executable_blake3(&self) -> &str { &self.host_executable_blake3 }
    pub const fn host_device(&self) -> u64 { self.host_device }
    pub const fn host_inode(&self) -> u64 { self.host_inode }
    pub const fn host_file_size(&self) -> u64 { self.host_file_size }
    pub fn backend_id(&self) -> &str { &self.backend_id }
    pub const fn collected_at_ms(&self) -> u64 { self.collected_at_ms }
    pub const fn verified_at_ms(&self) -> u64 { self.verified_at_ms }
    pub const fn signature_verification_in_process(&self) -> bool { true }
    pub const fn external_checkquote_process_required(&self) -> bool { false }
    pub const fn running_host_backing_file_identity_observed(&self) -> bool { true }
    pub const fn mapped_memory_identity_established(&self) -> bool { false }
    pub const fn dependency_closure_atomically_pinned(&self) -> bool { false }
    pub const fn privileged_in_place_mutation_excluded(&self) -> bool { false }
    pub const fn trusted_time_established(&self) -> bool { false }
    pub const fn root_resistant_immutability_established(&self) -> bool { false }
    pub const fn grants_physical_authority(&self) -> bool { false }
}

#[derive(Debug)]
pub struct InProcessTpm2QuoteQualification {
    pub report: InProcessTpm2EcdsaReport,
    pub parent: Tpm2QuoteQualification,
    verified: VerifiedInProcessTpm2Quote,
}

impl InProcessTpm2QuoteQualification {
    pub fn verified(&self) -> &VerifiedInProcessTpm2Quote { &self.verified }
    pub fn into_verified(self) -> VerifiedInProcessTpm2Quote { self.verified }
}

#[derive(Debug, Clone)]
struct InProcessEcdsaExecutor {
    host: RunningHostExecutableIdentity,
}

impl CheckquoteExecutor for InProcessEcdsaExecutor {
    fn executable_blake3(&self, executable: &str) -> Result<String, String> {
        if executable != self.host.canonical_path {
            return Err("parent verifier path does not name reviewed running host".into());
        }
        Ok(self.host.executable_blake3.clone())
    }

    fn execute_checkquote(
        &self,
        executable: &str,
        request: &CheckquoteExecutionRequest,
    ) -> Result<ToolExecution, String> {
        if executable != self.host.canonical_path {
            return Err("parent verifier path changed before in-process verification".into());
        }
        let accepted = verify_ecdsa_p256_sha256_request(request).is_ok();
        Ok(ToolExecution {
            exit_code: Some(if accepted { 0 } else { 1 }),
            stdout: Vec::new(),
            stderr: Vec::new(),
        })
    }
}

pub fn verify_tpm2_quote_in_process(
    policy: &InProcessTpm2EcdsaPolicy,
    parent_policy: &Tpm2CheckquotePolicy,
    challenge: &AttestationChallenge,
    ak: &AttestationKeyBinding,
    bundle: &RawTpm2QuoteBundle,
    verified_at_ms: u64,
) -> Result<InProcessTpm2QuoteQualification, InProcessTpm2EcdsaReport> {
    let policy_digest = policy.canonical_digest();
    let parent_policy_digest = parent_policy.canonical_digest();
    let mut report = InProcessTpm2EcdsaReport {
        schema_version: IN_PROCESS_TPM2_ECDSA_REPORT_SCHEMA_V1.into(),
        policy_id: policy.policy_id.clone(),
        policy_digest: policy_digest.clone(),
        parent_policy_digest: parent_policy_digest.clone(),
        parent_report_digest: None,
        parent_qualification_digest: None,
        host_identity_digest: None,
        host_executable_path: policy.expected_host_executable_path.clone(),
        host_nix_store_root: direct_nix_store_root(&policy.expected_host_executable_path)
            .unwrap_or_default(),
        host_executable_blake3: policy.expected_host_executable_blake3.clone(),
        backend_id: policy.expected_backend_id.clone(),
        disposition: InProcessTpm2EcdsaDisposition::Invalid,
        issues: Vec::new(),
    };

    if policy_digest.is_none() {
        report.issues.push(InProcessTpm2EcdsaIssue::InvalidPolicy);
        return Err(finalize(report));
    }
    if !parent_policy.validate() || parent_policy_digest.is_none() {
        report.issues.push(InProcessTpm2EcdsaIssue::InvalidParentPolicy);
        return Err(finalize(report));
    }
    if parent_policy_digest.as_deref() != Some(policy.expected_parent_policy_digest.as_str()) {
        report.issues.push(InProcessTpm2EcdsaIssue::ParentPolicyDigestMismatch);
        return Err(finalize(report));
    }
    if parent_policy.checkquote_path != policy.expected_host_executable_path {
        report.issues.push(InProcessTpm2EcdsaIssue::ParentVerifierPathMismatch);
        return Err(finalize(report));
    }
    if parent_policy.expected_checkquote_blake3 != policy.expected_host_executable_blake3 {
        report.issues.push(InProcessTpm2EcdsaIssue::ParentVerifierDigestMismatch);
        return Err(finalize(report));
    }
    if parent_policy.hash_algorithm != "sha256" {
        report.issues.push(InProcessTpm2EcdsaIssue::UnsupportedParentHashAlgorithm);
        return Err(finalize(report));
    }
    if ak.signing_scheme != SUPPORTED_AK_SIGNING_SCHEME_V1 {
        report.issues.push(InProcessTpm2EcdsaIssue::UnsupportedAkSigningScheme);
        return Err(finalize(report));
    }
    if extract_p256_spki_point(&bundle.ak_public).is_err() {
        report.issues.push(InProcessTpm2EcdsaIssue::UnsupportedAkPublicEncoding);
        return Err(finalize(report));
    }

    let host = match observe_running_host(policy) {
        Ok(value) => value,
        Err(HostObservationError::Unavailable(reason)) => {
            report.issues.push(InProcessTpm2EcdsaIssue::HostIdentityUnavailable(reason));
            return Err(finalize(report));
        }
        Err(HostObservationError::PathMismatch) => {
            report.issues.push(InProcessTpm2EcdsaIssue::HostPathMismatch);
            return Err(finalize(report));
        }
        Err(HostObservationError::DigestMismatch) => {
            report.issues.push(InProcessTpm2EcdsaIssue::HostExecutableDigestMismatch);
            return Err(finalize(report));
        }
        Err(HostObservationError::MetadataMismatch) => {
            report.issues.push(InProcessTpm2EcdsaIssue::HostMetadataMismatch);
            return Err(finalize(report));
        }
    };
    let host_identity_digest = host
        .canonical_digest()
        .expect("observed production host identity is canonical");
    report.host_identity_digest = Some(host_identity_digest.clone());
    report.host_nix_store_root = host.nix_store_root.clone();
    report.host_executable_blake3 = host.executable_blake3.clone();

    let executor = InProcessEcdsaExecutor { host: host.clone() };
    let parent = match verify_tpm2_quote(
        parent_policy,
        challenge,
        ak,
        bundle,
        verified_at_ms,
        &executor,
    ) {
        Ok(value) => value,
        Err(parent_report) => {
            report.parent_report_digest = Some(parent_report.canonical_digest());
            report.issues.push(match parent_report.disposition {
                Tpm2CheckquoteDisposition::Blocked => InProcessTpm2EcdsaIssue::ParentVerificationBlocked,
                _ => InProcessTpm2EcdsaIssue::ParentVerificationInvalid,
            });
            return Err(finalize(report));
        }
    };

    let parent_verified = parent.verified();
    report.parent_report_digest = Some(parent.report.canonical_digest());
    report.parent_qualification_digest = Some(parent_verified.qualification_digest().into());
    report.disposition = InProcessTpm2EcdsaDisposition::Qualified;
    let report_digest = report.canonical_digest();
    let policy_digest = policy_digest.expect("validated policy has digest");
    let qualification_digest = qualification_digest(
        &policy_digest,
        &policy.expected_parent_policy_digest,
        parent_verified.qualification_digest(),
        &host_identity_digest,
        IN_PROCESS_BACKEND_ID_V1,
        &report_digest,
    );
    let verified = VerifiedInProcessTpm2Quote {
        qualification_digest,
        report_digest,
        policy_digest,
        parent_policy_digest: policy.expected_parent_policy_digest.clone(),
        parent_qualification_digest: parent_verified.qualification_digest().into(),
        challenge_digest: parent_verified.challenge_digest().into(),
        ak_binding_digest: parent_verified.ak_binding_digest().into(),
        quote_artifact_digest: parent_verified.quote_artifact_digest().into(),
        verification_receipt_digest: parent_verified.verification_receipt_digest().into(),
        pcr_selection_digest: parent_verified.pcr_selection_digest().into(),
        host_identity_digest,
        host_executable_path: host.canonical_path,
        host_nix_store_root: host.nix_store_root,
        host_executable_blake3: host.executable_blake3,
        host_device: host.device,
        host_inode: host.inode,
        host_file_size: host.file_size,
        backend_id: IN_PROCESS_BACKEND_ID_V1.into(),
        collected_at_ms: parent_verified.collected_at_ms(),
        verified_at_ms: parent_verified.verified_at_ms(),
    };

    Ok(InProcessTpm2QuoteQualification { report, parent, verified })
}

fn finalize(mut report: InProcessTpm2EcdsaReport) -> InProcessTpm2EcdsaReport {
    report.disposition = if report.issues.iter().any(InProcessTpm2EcdsaIssue::is_invalid) {
        InProcessTpm2EcdsaDisposition::Invalid
    } else {
        InProcessTpm2EcdsaDisposition::Blocked
    };
    report
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum CryptoRequestError {
    UnsupportedPublicKey,
    MalformedSignature,
    SignatureRejected,
}

fn verify_ecdsa_p256_sha256_request(
    request: &CheckquoteExecutionRequest,
) -> Result<(), CryptoRequestError> {
    let point = extract_p256_spki_point(&request.ak_public)
        .map_err(|_| CryptoRequestError::UnsupportedPublicKey)?;
    let signature = parse_tss_ecdsa_sha256_signature(&request.signature)
        .map_err(|_| CryptoRequestError::MalformedSignature)?;
    UnparsedPublicKey::new(&ECDSA_P256_SHA256_FIXED, point)
        .verify(&request.quote_message, &signature)
        .map_err(|_| CryptoRequestError::SignatureRejected)
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum PublicKeyParseError {
    Encoding,
    Algorithm,
}

fn extract_p256_spki_point(input: &[u8]) -> Result<Vec<u8>, PublicKeyParseError> {
    let der = decode_public_key_der(input)?;
    let mut outer = DerCursor::new(&der);
    let sequence = outer.tlv(0x30)?;
    if !outer.done() {
        return Err(PublicKeyParseError::Encoding);
    }
    let mut spki = DerCursor::new(sequence);
    let algorithm_sequence = spki.tlv(0x30)?;
    let bit_string = spki.tlv(0x03)?;
    if !spki.done() {
        return Err(PublicKeyParseError::Encoding);
    }
    let mut algorithm = DerCursor::new(algorithm_sequence);
    let algorithm_oid = algorithm.tlv(0x06)?;
    let curve_oid = algorithm.tlv(0x06)?;
    if !algorithm.done() || algorithm_oid != EC_PUBLIC_KEY_OID || curve_oid != PRIME256V1_OID {
        return Err(PublicKeyParseError::Algorithm);
    }
    if bit_string.len() != 66 || bit_string[0] != 0 || bit_string[1] != 0x04 {
        return Err(PublicKeyParseError::Encoding);
    }
    Ok(bit_string[1..].to_vec())
}

fn decode_public_key_der(input: &[u8]) -> Result<Vec<u8>, PublicKeyParseError> {
    const BEGIN: &str = "-----BEGIN PUBLIC KEY-----";
    const END: &str = "-----END PUBLIC KEY-----";
    if !input.starts_with(BEGIN.as_bytes()) {
        return Ok(input.to_vec());
    }
    let text = std::str::from_utf8(input).map_err(|_| PublicKeyParseError::Encoding)?;
    let lines = text.lines().collect::<Vec<_>>();
    if lines.len() < 3 || lines.first() != Some(&BEGIN) || lines.last() != Some(&END) {
        return Err(PublicKeyParseError::Encoding);
    }
    let body = &lines[1..lines.len() - 1];
    if body.iter().any(|line| line.is_empty() || line.chars().any(char::is_whitespace)) {
        return Err(PublicKeyParseError::Encoding);
    }
    base64::engine::general_purpose::STANDARD
        .decode(body.concat())
        .map_err(|_| PublicKeyParseError::Encoding)
}

struct DerCursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> DerCursor<'a> {
    fn new(bytes: &'a [u8]) -> Self { Self { bytes, pos: 0 } }
    fn done(&self) -> bool { self.pos == self.bytes.len() }

    fn tlv(&mut self, tag: u8) -> Result<&'a [u8], PublicKeyParseError> {
        if self.pos >= self.bytes.len() || self.bytes[self.pos] != tag {
            return Err(PublicKeyParseError::Encoding);
        }
        self.pos += 1;
        let len = self.length()?;
        let end = self.pos.checked_add(len).ok_or(PublicKeyParseError::Encoding)?;
        if end > self.bytes.len() {
            return Err(PublicKeyParseError::Encoding);
        }
        let value = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(value)
    }

    fn length(&mut self) -> Result<usize, PublicKeyParseError> {
        let first = *self.bytes.get(self.pos).ok_or(PublicKeyParseError::Encoding)?;
        self.pos += 1;
        if first & 0x80 == 0 {
            return Ok(first as usize);
        }
        let count = (first & 0x7f) as usize;
        if count == 0 || count > std::mem::size_of::<usize>() {
            return Err(PublicKeyParseError::Encoding);
        }
        if self.pos + count > self.bytes.len() || self.bytes[self.pos] == 0 {
            return Err(PublicKeyParseError::Encoding);
        }
        let mut len = 0usize;
        for byte in &self.bytes[self.pos..self.pos + count] {
            len = len.checked_mul(256).and_then(|v| v.checked_add(*byte as usize))
                .ok_or(PublicKeyParseError::Encoding)?;
        }
        self.pos += count;
        if len < 128 {
            return Err(PublicKeyParseError::Encoding);
        }
        Ok(len)
    }
}

fn parse_tss_ecdsa_sha256_signature(input: &[u8]) -> Result<[u8; 64], ()> {
    let mut cursor = WireCursor::new(input);
    if cursor.u16()? != TPM2_ALG_ECDSA || cursor.u16()? != TPM2_ALG_SHA256 {
        return Err(());
    }
    let r = cursor.tpm2b()?;
    let s = cursor.tpm2b()?;
    if !cursor.done() || r.is_empty() || s.is_empty() || r.len() > 32 || s.len() > 32 {
        return Err(());
    }
    let mut fixed = [0u8; 64];
    fixed[32 - r.len()..32].copy_from_slice(r);
    fixed[64 - s.len()..64].copy_from_slice(s);
    Ok(fixed)
}

struct WireCursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> WireCursor<'a> {
    fn new(bytes: &'a [u8]) -> Self { Self { bytes, pos: 0 } }
    fn done(&self) -> bool { self.pos == self.bytes.len() }
    fn u16(&mut self) -> Result<u16, ()> {
        let raw = self.bytes.get(self.pos..self.pos + 2).ok_or(())?;
        self.pos += 2;
        Ok(u16::from_be_bytes([raw[0], raw[1]]))
    }
    fn tpm2b(&mut self) -> Result<&'a [u8], ()> {
        let len = self.u16()? as usize;
        let end = self.pos.checked_add(len).ok_or(())?;
        let value = self.bytes.get(self.pos..end).ok_or(())?;
        self.pos = end;
        Ok(value)
    }
}

#[derive(Debug)]
enum HostObservationError {
    Unavailable(String),
    PathMismatch,
    DigestMismatch,
    MetadataMismatch,
}

#[cfg(target_os = "linux")]
fn observe_running_host(
    policy: &InProcessTpm2EcdsaPolicy,
) -> Result<RunningHostExecutableIdentity, HostObservationError> {
    use std::fs::File;
    use std::io::Read;
    use std::os::unix::fs::MetadataExt;

    let expected = PathBuf::from(&policy.expected_host_executable_path);
    let canonical_expected = std::fs::canonicalize(&expected)
        .map_err(|e| HostObservationError::Unavailable(format!("canonicalize expected host: {e}")))?;
    if canonical_expected != expected {
        return Err(HostObservationError::PathMismatch);
    }
    let observed_path = std::fs::read_link("/proc/self/exe")
        .map_err(|e| HostObservationError::Unavailable(format!("read /proc/self/exe: {e}")))?;
    if observed_path != expected {
        return Err(HostObservationError::PathMismatch);
    }
    let store_root = direct_nix_store_root(&policy.expected_host_executable_path)
        .ok_or(HostObservationError::PathMismatch)?;
    let root_meta = std::fs::metadata(&store_root)
        .map_err(|e| HostObservationError::Unavailable(format!("stat store root: {e}")))?;
    if root_meta.uid() != 0 || root_meta.mode() & 0o222 != 0 {
        return Err(HostObservationError::MetadataMismatch);
    }

    let expected_meta = std::fs::metadata(&expected)
        .map_err(|e| HostObservationError::Unavailable(format!("stat expected host: {e}")))?;
    if expected_meta.uid() != 0
        || expected_meta.mode() & 0o222 != 0
        || expected_meta.mode() & 0o111 == 0
        || expected_meta.len() == 0
        || expected_meta.len() > policy.max_host_executable_bytes
    {
        return Err(HostObservationError::MetadataMismatch);
    }

    let mut file = File::open("/proc/self/exe")
        .map_err(|e| HostObservationError::Unavailable(format!("open /proc/self/exe: {e}")))?;
    let before = file.metadata()
        .map_err(|e| HostObservationError::Unavailable(format!("fstat running host: {e}")))?;
    if before.dev() != expected_meta.dev()
        || before.ino() != expected_meta.ino()
        || before.len() != expected_meta.len()
        || before.mode() != expected_meta.mode()
        || before.uid() != expected_meta.uid()
        || before.gid() != expected_meta.gid()
    {
        return Err(HostObservationError::MetadataMismatch);
    }
    let mut hasher = blake3::Hasher::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = file.read(&mut buffer)
            .map_err(|e| HostObservationError::Unavailable(format!("read running host: {e}")))?;
        if read == 0 { break; }
        hasher.update(&buffer[..read]);
    }
    let after = file.metadata()
        .map_err(|e| HostObservationError::Unavailable(format!("postflight fstat running host: {e}")))?;
    if before.dev() != after.dev()
        || before.ino() != after.ino()
        || before.len() != after.len()
        || before.mode() != after.mode()
        || before.uid() != after.uid()
        || before.gid() != after.gid()
        || before.mtime() != after.mtime()
        || before.mtime_nsec() != after.mtime_nsec()
    {
        return Err(HostObservationError::MetadataMismatch);
    }
    let executable_blake3 = format!("blake3:{}", hasher.finalize().to_hex());
    if executable_blake3 != policy.expected_host_executable_blake3 {
        return Err(HostObservationError::DigestMismatch);
    }
    Ok(RunningHostExecutableIdentity {
        canonical_path: policy.expected_host_executable_path.clone(),
        nix_store_root: store_root,
        executable_blake3,
        device: before.dev(),
        inode: before.ino(),
        file_size: before.len(),
        mode: before.mode(),
        uid: before.uid(),
        gid: before.gid(),
        mtime_seconds: before.mtime(),
        mtime_nanoseconds: before.mtime_nsec(),
    })
}

#[cfg(not(target_os = "linux"))]
fn observe_running_host(
    _policy: &InProcessTpm2EcdsaPolicy,
) -> Result<RunningHostExecutableIdentity, HostObservationError> {
    Err(HostObservationError::Unavailable(
        "Linux /proc/self/exe identity is required".into(),
    ))
}

fn qualification_digest(
    policy_digest: &str,
    parent_policy_digest: &str,
    parent_qualification_digest: &str,
    host_identity_digest: &str,
    backend_id: &str,
    report_digest: &str,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(QUALIFICATION_DIGEST_DOMAIN);
    for field in [
        policy_digest,
        parent_policy_digest,
        parent_qualification_digest,
        host_identity_digest,
        backend_id,
        report_digest,
    ] {
        push_field(&mut hasher, field);
    }
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn direct_nix_store_root(path: &str) -> Option<String> {
    let suffix = path.strip_prefix("/nix/store/")?;
    let slash = suffix.find('/')?;
    let component = &suffix[..slash];
    if component.len() <= 33 || component.as_bytes()[32] != b'-' {
        return None;
    }
    if !component.as_bytes()[..32].iter().all(|b| NIX_BASE32.contains(b)) {
        return None;
    }
    if component[33..].is_empty()
        || !component[33..].bytes().all(|b| {
            b.is_ascii_alphanumeric() || matches!(b, b'+' | b'-' | b'.' | b'_' | b'?' | b'=')
        })
    {
        return None;
    }
    if Path::new(path).components().any(|part| matches!(part, std::path::Component::ParentDir)) {
        return None;
    }
    Some(format!("/nix/store/{component}"))
}

fn canonical_text(value: &str) -> bool {
    !value.is_empty()
        && value == value.trim()
        && value.len() <= MAX_TEXT_BYTES
        && !value.chars().any(char::is_control)
}

fn valid_blake3_digest(value: &str) -> bool {
    let Some(hex) = value.strip_prefix("blake3:") else { return false; };
    hex.len() == 64 && hex.bytes().all(|b| b.is_ascii_hexdigit() && !b.is_ascii_uppercase())
}

fn valid_refs(refs: &[String]) -> bool {
    if refs.len() > MAX_EVIDENCE_REFS || !refs.iter().all(|value| canonical_text(value)) {
        return false;
    }
    let mut seen = BTreeSet::new();
    refs.iter().all(|value| seen.insert(value.as_str()))
}

fn push_field(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn push_sorted_refs(hasher: &mut blake3::Hasher, refs: &[String]) {
    let mut refs = refs.to_vec();
    refs.sort();
    hasher.update(&(refs.len() as u64).to_le_bytes());
    for reference in refs { push_field(hasher, &reference); }
}

#[cfg(test)]
mod tests {
    use super::*;
    use aws_lc_rs::rand::SystemRandom;
    use aws_lc_rs::signature::{EcdsaKeyPair, KeyPair, ECDSA_P256_SHA256_FIXED_SIGNING};

    fn blake(value: &[u8]) -> String {
        format!("blake3:{}", blake3::hash(value).to_hex())
    }

    fn policy() -> InProcessTpm2EcdsaPolicy {
        InProcessTpm2EcdsaPolicy {
            schema_version: IN_PROCESS_TPM2_ECDSA_POLICY_SCHEMA_V1.into(),
            policy_id: "policy:in-process-tpm2-ecdsa:1".into(),
            expected_parent_policy_digest: blake(b"parent-policy"),
            expected_host_executable_path:
                "/nix/store/0123456789abcdfghijklmnpqrsvwxyz-symthaea/bin/symthaea".into(),
            expected_host_executable_blake3: blake(b"host"),
            expected_backend_id: IN_PROCESS_BACKEND_ID_V1.into(),
            max_host_executable_bytes: 64 * 1024 * 1024,
            evidence_refs: vec!["review:host".into(), "review:crypto".into()],
        }
    }

    fn der_wrap(tag: u8, value: &[u8]) -> Vec<u8> {
        let mut out = vec![tag];
        if value.len() < 128 {
            out.push(value.len() as u8);
        } else {
            out.push(0x81);
            out.push(value.len() as u8);
        }
        out.extend_from_slice(value);
        out
    }

    fn spki(point: &[u8]) -> Vec<u8> {
        let mut algorithm = der_wrap(0x06, EC_PUBLIC_KEY_OID);
        algorithm.extend_from_slice(&der_wrap(0x06, PRIME256V1_OID));
        let algorithm = der_wrap(0x30, &algorithm);
        let mut bits = vec![0u8];
        bits.extend_from_slice(point);
        let mut body = algorithm;
        body.extend_from_slice(&der_wrap(0x03, &bits));
        der_wrap(0x30, &body)
    }

    fn pem(der: &[u8]) -> Vec<u8> {
        let encoded = base64::engine::general_purpose::STANDARD.encode(der);
        format!("-----BEGIN PUBLIC KEY-----\n{encoded}\n-----END PUBLIC KEY-----").into_bytes()
    }

    fn tss_signature(fixed: &[u8]) -> Vec<u8> {
        assert_eq!(fixed.len(), 64);
        let mut out = Vec::new();
        out.extend_from_slice(&TPM2_ALG_ECDSA.to_be_bytes());
        out.extend_from_slice(&TPM2_ALG_SHA256.to_be_bytes());
        out.extend_from_slice(&32u16.to_be_bytes());
        out.extend_from_slice(&fixed[..32]);
        out.extend_from_slice(&32u16.to_be_bytes());
        out.extend_from_slice(&fixed[32..]);
        out
    }

    #[test]
    fn policy_reference_order_is_nonsemantic() {
        let left = policy();
        let mut right = left.clone();
        right.evidence_refs.reverse();
        assert_eq!(left.canonical_digest(), right.canonical_digest());
        right.expected_backend_id = "other-backend".into();
        assert_eq!(right.canonical_digest(), None);
    }

    #[test]
    fn p256_spki_der_and_pem_extract_same_point() {
        let key = EcdsaKeyPair::generate(&ECDSA_P256_SHA256_FIXED_SIGNING).unwrap();
        let point = key.public_key().as_ref();
        let der = spki(point);
        assert_eq!(extract_p256_spki_point(&der).unwrap(), point);
        assert_eq!(extract_p256_spki_point(&pem(&der)).unwrap(), point);
    }

    #[test]
    fn in_process_tss_ecdsa_signature_verifies() {
        let key = EcdsaKeyPair::generate(&ECDSA_P256_SHA256_FIXED_SIGNING).unwrap();
        let message = b"exact TPMS_ATTEST bytes";
        let signature = key.sign(&SystemRandom::new(), message).unwrap();
        let der = spki(key.public_key().as_ref());
        let request = CheckquoteExecutionRequest {
            ak_public: pem(&der),
            quote_message: message.to_vec(),
            signature: tss_signature(signature.as_ref()),
            pcr_values_raw: Vec::new(),
            pcr_list: "sha256:10".into(),
            qualification_hex: "00".repeat(32),
            hash_algorithm: "sha256".into(),
        };
        assert_eq!(verify_ecdsa_p256_sha256_request(&request), Ok(()));
    }

    #[test]
    fn modified_signature_is_rejected() {
        let key = EcdsaKeyPair::generate(&ECDSA_P256_SHA256_FIXED_SIGNING).unwrap();
        let message = b"exact TPMS_ATTEST bytes";
        let signature = key.sign(&SystemRandom::new(), message).unwrap();
        let mut tss = tss_signature(signature.as_ref());
        let last = tss.len() - 1;
        tss[last] ^= 0x01;
        let request = CheckquoteExecutionRequest {
            ak_public: spki(key.public_key().as_ref()),
            quote_message: message.to_vec(),
            signature: tss,
            pcr_values_raw: Vec::new(),
            pcr_list: "sha256:10".into(),
            qualification_hex: "00".repeat(32),
            hash_algorithm: "sha256".into(),
        };
        assert_eq!(
            verify_ecdsa_p256_sha256_request(&request),
            Err(CryptoRequestError::SignatureRejected)
        );
    }

    #[test]
    fn wrong_curve_oid_is_rejected() {
        let key = EcdsaKeyPair::generate(&ECDSA_P256_SHA256_FIXED_SIGNING).unwrap();
        let mut der = spki(key.public_key().as_ref());
        let needle = PRIME256V1_OID;
        let at = der.windows(needle.len()).position(|window| window == needle).unwrap();
        der[at + needle.len() - 1] ^= 1;
        assert!(extract_p256_spki_point(&der).is_err());
    }

    #[test]
    fn signature_trailing_bytes_are_rejected() {
        let mut signature = vec![0u8; 4 + 2 + 32 + 2 + 32];
        signature[0..2].copy_from_slice(&TPM2_ALG_ECDSA.to_be_bytes());
        signature[2..4].copy_from_slice(&TPM2_ALG_SHA256.to_be_bytes());
        signature[4..6].copy_from_slice(&32u16.to_be_bytes());
        signature[38..40].copy_from_slice(&32u16.to_be_bytes());
        signature.push(0);
        assert!(parse_tss_ecdsa_sha256_signature(&signature).is_err());
    }

    #[test]
    fn oversized_signature_component_is_rejected() {
        let mut signature = Vec::new();
        signature.extend_from_slice(&TPM2_ALG_ECDSA.to_be_bytes());
        signature.extend_from_slice(&TPM2_ALG_SHA256.to_be_bytes());
        signature.extend_from_slice(&33u16.to_be_bytes());
        signature.extend_from_slice(&[0u8; 33]);
        signature.extend_from_slice(&1u16.to_be_bytes());
        signature.push(1);
        assert!(parse_tss_ecdsa_sha256_signature(&signature).is_err());
    }
}
