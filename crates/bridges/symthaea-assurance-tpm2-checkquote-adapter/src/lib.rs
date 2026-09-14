// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Raw TPM2 quote verification with independent TPMS_ATTEST validation.
//!
//! Cryptographic signature verification is delegated to an exact-digest-pinned
//! `tpm2_checkquote` binary. Symthaea separately parses the signed attestation
//! bytes and validates magic, type, nonce, qualified signer, PCR selection and
//! PCR composite before minting the existing semantic verification receipt.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::path::Path;
use symthaea_assurance_tpm2_attestation_possession::{
    pcr_selection_digest, AttestationChallenge, AttestationKeyBinding, PcrBankSelection,
    QuoteVerificationReceipt, Tpm2QuoteArtifacts,
};

pub const CHECKQUOTE_POLICY_SCHEMA_V1: &str =
    "symthaea.assurance.tpm2-checkquote-adapter-policy.v1";
pub const CHECKQUOTE_REPORT_SCHEMA_V1: &str =
    "symthaea.assurance.tpm2-checkquote-adapter-report.v1";

const POLICY_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.tpm2-checkquote-policy.digest.v1\0";
const REPORT_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.tpm2-checkquote-report.digest.v1\0";
const QUALIFICATION_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.tpm2-checkquote-qualification.digest.v1\0";
const TPM2_GENERATED_VALUE: u32 = 0xff54_4347;
const TPM2_ST_ATTEST_QUOTE: u16 = 0x8018;
const TPM2_ALG_SHA256: u16 = 0x000b;
const SHA256_LEN: usize = 32;
const MAX_TEXT_BYTES: usize = 512;
const MAX_EVIDENCE_REFS: usize = 128;
const MAX_NAME_BYTES: usize = 256;
const MAX_EXTRA_DATA_BYTES: usize = 1024;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tpm2CheckquotePolicy {
    pub schema_version: String,
    pub adapter_id: String,
    pub verifier_ref: String,
    pub checkquote_path: String,
    pub expected_checkquote_blake3: String,
    /// First tranche is intentionally SHA-256 only.
    pub hash_algorithm: String,
    pub max_public_bytes: u64,
    pub max_message_bytes: u64,
    pub max_signature_bytes: u64,
    pub max_tool_output_bytes: u64,
    pub evidence_refs: Vec<String>,
}

impl Tpm2CheckquotePolicy {
    pub fn validate(&self) -> bool {
        self.schema_version == CHECKQUOTE_POLICY_SCHEMA_V1
            && canonical_text(&self.adapter_id)
            && canonical_text(&self.verifier_ref)
            && Path::new(&self.checkquote_path).is_absolute()
            && valid_blake3_digest(&self.expected_checkquote_blake3)
            && self.hash_algorithm == "sha256"
            && valid_limit(self.max_public_bytes, 16 * 1024 * 1024)
            && valid_limit(self.max_message_bytes, 1024 * 1024)
            && valid_limit(self.max_signature_bytes, 1024 * 1024)
            && valid_limit(self.max_tool_output_bytes, 1024 * 1024)
            && valid_refs(&self.evidence_refs)
    }

    pub fn canonical_digest(&self) -> Option<String> {
        if !self.validate() {
            return None;
        }
        let mut hasher = blake3::Hasher::new();
        hasher.update(POLICY_DIGEST_DOMAIN);
        for value in [
            self.schema_version.as_str(),
            self.adapter_id.as_str(),
            self.verifier_ref.as_str(),
            self.checkquote_path.as_str(),
            self.expected_checkquote_blake3.as_str(),
            self.hash_algorithm.as_str(),
        ] {
            push_field(&mut hasher, value);
        }
        for value in [
            self.max_public_bytes,
            self.max_message_bytes,
            self.max_signature_bytes,
            self.max_tool_output_bytes,
        ] {
            hasher.update(&value.to_le_bytes());
        }
        push_sorted_refs(&mut hasher, &self.evidence_refs);
        Some(format!("blake3:{}", hasher.finalize().to_hex()))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Sha256PcrValue {
    pub pcr: u8,
    pub value: [u8; SHA256_LEN],
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct RawTpm2QuoteBundle {
    pub quote_id: String,
    pub platform_qualification_digest: String,
    /// PEM or TSS public-key bytes accepted by the pinned checkquote tool.
    pub ak_public: Vec<u8>,
    /// Exact raw TPMS_ATTEST bytes emitted by `tpm2_quote -m`.
    pub quote_message: Vec<u8>,
    /// Exact signature bytes emitted by `tpm2_quote -s`.
    pub signature: Vec<u8>,
    /// Structured SHA-256 PCR values in strictly increasing PCR order.
    pub pcr_values: Vec<Sha256PcrValue>,
    pub collected_at_ms: u64,
    pub evidence_refs: Vec<String>,
}

impl RawTpm2QuoteBundle {
    fn validate(&self, policy: &Tpm2CheckquotePolicy) -> bool {
        canonical_text(&self.quote_id)
            && valid_blake3_digest(&self.platform_qualification_digest)
            && !self.ak_public.is_empty()
            && self.ak_public.len() as u64 <= policy.max_public_bytes
            && !self.quote_message.is_empty()
            && self.quote_message.len() as u64 <= policy.max_message_bytes
            && !self.signature.is_empty()
            && self.signature.len() as u64 <= policy.max_signature_bytes
            && !self.pcr_values.is_empty()
            && self.pcr_values.len() <= 24
            && self.pcr_values.iter().all(|value| value.pcr <= 23)
            && self
                .pcr_values
                .windows(2)
                .all(|pair| pair[0].pcr < pair[1].pcr)
            && self.collected_at_ms > 0
            && valid_refs(&self.evidence_refs)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckquoteExecutionRequest {
    pub ak_public: Vec<u8>,
    pub quote_message: Vec<u8>,
    pub signature: Vec<u8>,
    /// Normalized mode: selected digest bytes only, exact selection comes from `pcr_list`.
    pub pcr_values_raw: Vec<u8>,
    pub pcr_list: String,
    pub qualification_hex: String,
    pub hash_algorithm: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolExecution {
    pub exit_code: Option<i32>,
    pub stdout: Vec<u8>,
    pub stderr: Vec<u8>,
}

pub trait CheckquoteExecutor {
    fn executable_blake3(&self, executable: &str) -> Result<String, String>;

    fn execute_checkquote(
        &self,
        executable: &str,
        request: &CheckquoteExecutionRequest,
    ) -> Result<ToolExecution, String>;
}

#[cfg(target_os = "linux")]
#[derive(Debug, Default, Clone, Copy)]
pub struct SystemTpm2CheckquoteExecutor;

#[cfg(target_os = "linux")]
impl CheckquoteExecutor for SystemTpm2CheckquoteExecutor {
    fn executable_blake3(&self, executable: &str) -> Result<String, String> {
        hash_file_blake3(executable)
    }

    fn execute_checkquote(
        &self,
        executable: &str,
        request: &CheckquoteExecutionRequest,
    ) -> Result<ToolExecution, String> {
        use std::process::{Command, Stdio};

        let workspace = SecureTempWorkspace::create()?;
        let public_path = workspace.write("ak-public.bin", &request.ak_public)?;
        let message_path = workspace.write("quote-message.bin", &request.quote_message)?;
        let signature_path = workspace.write("quote-signature.bin", &request.signature)?;
        let pcr_path = workspace.write("pcr-values.bin", &request.pcr_values_raw)?;

        let output = Command::new(executable)
            .env_clear()
            .arg("-u")
            .arg(public_path)
            .arg("-m")
            .arg(message_path)
            .arg("-s")
            .arg(signature_path)
            .arg("-f")
            .arg(pcr_path)
            .arg("-l")
            .arg(&request.pcr_list)
            .arg("-g")
            .arg(&request.hash_algorithm)
            .arg("-q")
            .arg(&request.qualification_hex)
            .stdin(Stdio::null())
            .output()
            .map_err(|error| format!("checkquote execution failed: {error}"))?;

        Ok(ToolExecution {
            exit_code: output.status.code(),
            stdout: output.stdout,
            stderr: output.stderr,
        })
    }
}

#[cfg(target_os = "linux")]
fn hash_file_blake3(path: &str) -> Result<String, String> {
    use std::fs::File;
    use std::io::Read;

    let mut file = File::open(path).map_err(|error| format!("open executable: {error}"))?;
    let mut hasher = blake3::Hasher::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let read = file
            .read(&mut buffer)
            .map_err(|error| format!("read executable: {error}"))?;
        if read == 0 {
            break;
        }
        hasher.update(&buffer[..read]);
    }
    Ok(format!("blake3:{}", hasher.finalize().to_hex()))
}

#[cfg(target_os = "linux")]
struct SecureTempWorkspace {
    path: std::path::PathBuf,
}

#[cfg(target_os = "linux")]
impl SecureTempWorkspace {
    fn create() -> Result<Self, String> {
        use std::os::unix::fs::DirBuilderExt;
        use std::sync::atomic::{AtomicU64, Ordering};
        use std::time::{SystemTime, UNIX_EPOCH};

        static COUNTER: AtomicU64 = AtomicU64::new(1);
        let base = std::env::temp_dir();
        for _ in 0..32 {
            let nonce = COUNTER.fetch_add(1, Ordering::Relaxed);
            let now = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos();
            let path = base.join(format!(
                ".symthaea-tpm2-checkquote-{}-{now}-{nonce}",
                std::process::id()
            ));
            let mut builder = std::fs::DirBuilder::new();
            builder.mode(0o700);
            match builder.create(&path) {
                Ok(()) => return Ok(Self { path }),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => return Err(format!("create private temp dir: {error}")),
            }
        }
        Err("could not create unique private temp dir".into())
    }

    fn write(&self, name: &str, bytes: &[u8]) -> Result<std::path::PathBuf, String> {
        use std::io::Write;
        use std::os::unix::fs::OpenOptionsExt;

        let path = self.path.join(name);
        let mut file = std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&path)
            .map_err(|error| format!("create private temp file: {error}"))?;
        file.write_all(bytes)
            .map_err(|error| format!("write private temp file: {error}"))?;
        file.sync_all()
            .map_err(|error| format!("sync private temp file: {error}"))?;
        Ok(path)
    }
}

#[cfg(target_os = "linux")]
impl Drop for SecureTempWorkspace {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.path);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Tpm2CheckquoteDisposition {
    Invalid,
    Blocked,
    Qualified,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Tpm2CheckquoteIssue {
    InvalidPolicy,
    InvalidChallenge,
    ChallengeVerifierMismatch,
    InvalidAkBinding,
    UnsupportedAkNameAlgorithm,
    InvalidQuoteBundle,
    QuoteOutsideChallengeWindow,
    VerificationBeforeCollection,
    UnsupportedPcrSelection,
    PcrValuesDoNotMatchSelection,
    AkPublicDigestMismatch,
    QuoteTruncated(String),
    QuoteTrailingBytes(usize),
    AttestationMagicMismatch(u32),
    AttestationTypeMismatch(u16),
    QualifiedSignerMismatch,
    QualifyingDataMismatch,
    InvalidClockSafe(u8),
    PcrSelectionMismatch,
    PcrDigestLengthMismatch(usize),
    PcrCompositeMismatch,
    ToolDigestMismatch,
    ToolExecutionUnavailable(String),
    ToolRejected { exit_code: Option<i32> },
    ToolEmittedStderr,
    ToolOutputTooLarge { stream: String, observed: u64, maximum: u64 },
}

impl Tpm2CheckquoteIssue {
    fn is_invalid(&self) -> bool {
        !matches!(
            self,
            Self::ToolDigestMismatch | Self::ToolExecutionUnavailable(_)
        )
    }

    fn code(&self) -> String {
        match self {
            Self::InvalidPolicy => "invalid-policy".into(),
            Self::InvalidChallenge => "invalid-challenge".into(),
            Self::ChallengeVerifierMismatch => "challenge-verifier-mismatch".into(),
            Self::InvalidAkBinding => "invalid-ak-binding".into(),
            Self::UnsupportedAkNameAlgorithm => "unsupported-ak-name-algorithm".into(),
            Self::InvalidQuoteBundle => "invalid-quote-bundle".into(),
            Self::QuoteOutsideChallengeWindow => "quote-outside-challenge-window".into(),
            Self::VerificationBeforeCollection => "verification-before-collection".into(),
            Self::UnsupportedPcrSelection => "unsupported-pcr-selection".into(),
            Self::PcrValuesDoNotMatchSelection => "pcr-values-selection-mismatch".into(),
            Self::AkPublicDigestMismatch => "ak-public-digest-mismatch".into(),
            Self::QuoteTruncated(stage) => format!("quote-truncated:{stage}"),
            Self::QuoteTrailingBytes(count) => format!("quote-trailing-bytes:{count}"),
            Self::AttestationMagicMismatch(value) => format!("magic-mismatch:{value:#x}"),
            Self::AttestationTypeMismatch(value) => format!("type-mismatch:{value:#x}"),
            Self::QualifiedSignerMismatch => "qualified-signer-mismatch".into(),
            Self::QualifyingDataMismatch => "qualifying-data-mismatch".into(),
            Self::InvalidClockSafe(value) => format!("invalid-clock-safe:{value}"),
            Self::PcrSelectionMismatch => "pcr-selection-mismatch".into(),
            Self::PcrDigestLengthMismatch(value) => format!("pcr-digest-length:{value}"),
            Self::PcrCompositeMismatch => "pcr-composite-mismatch".into(),
            Self::ToolDigestMismatch => "tool-digest-mismatch".into(),
            Self::ToolExecutionUnavailable(value) => format!("tool-unavailable:{value}"),
            Self::ToolRejected { exit_code } => format!("tool-rejected:{exit_code:?}"),
            Self::ToolEmittedStderr => "tool-emitted-stderr".into(),
            Self::ToolOutputTooLarge { stream, observed, maximum } => {
                format!("tool-output-too-large:{stream}:{observed}:{maximum}")
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tpm2QuoteCheckReport {
    pub schema_version: String,
    pub adapter_id: String,
    pub policy_digest: Option<String>,
    pub challenge_digest: Option<String>,
    pub ak_binding_digest: Option<String>,
    pub quote_message_digest: String,
    pub signature_digest: String,
    pub pcr_values_digest: String,
    pub parsed_qualified_signer_hex: Option<String>,
    pub parsed_pcr_selection: Vec<PcrBankSelection>,
    pub parsed_pcr_digest_hex: Option<String>,
    pub quote_artifact_digest: Option<String>,
    pub verification_receipt_digest: Option<String>,
    pub verified_at_ms: u64,
    pub disposition: Tpm2CheckquoteDisposition,
    pub issues: Vec<Tpm2CheckquoteIssue>,
}

impl Tpm2QuoteCheckReport {
    pub fn canonical_digest(&self) -> String {
        let mut hasher = blake3::Hasher::new();
        hasher.update(REPORT_DIGEST_DOMAIN);
        for value in [
            self.schema_version.as_str(),
            self.adapter_id.as_str(),
            self.policy_digest.as_deref().unwrap_or("-"),
            self.challenge_digest.as_deref().unwrap_or("-"),
            self.ak_binding_digest.as_deref().unwrap_or("-"),
            self.quote_message_digest.as_str(),
            self.signature_digest.as_str(),
            self.pcr_values_digest.as_str(),
            self.parsed_qualified_signer_hex.as_deref().unwrap_or("-"),
            self.parsed_pcr_digest_hex.as_deref().unwrap_or("-"),
            self.quote_artifact_digest.as_deref().unwrap_or("-"),
            self.verification_receipt_digest.as_deref().unwrap_or("-"),
        ] {
            push_field(&mut hasher, value);
        }
        for bank in &self.parsed_pcr_selection {
            push_field(&mut hasher, &bank.hash_alg);
            hasher.update(&(bank.pcrs.len() as u64).to_le_bytes());
            hasher.update(&bank.pcrs);
        }
        hasher.update(&self.verified_at_ms.to_le_bytes());
        push_field(
            &mut hasher,
            match self.disposition {
                Tpm2CheckquoteDisposition::Invalid => "invalid",
                Tpm2CheckquoteDisposition::Blocked => "blocked",
                Tpm2CheckquoteDisposition::Qualified => "qualified",
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

/// Opaque, capability-bearing raw quote verification result.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedTpm2Quote {
    qualification_digest: String,
    report_digest: String,
    policy_digest: String,
    challenge_digest: String,
    ak_binding_digest: String,
    quote_artifact_digest: String,
    verification_receipt_digest: String,
    pcr_selection_digest: String,
    pcr_values: Vec<Sha256PcrValue>,
    collected_at_ms: u64,
    verified_at_ms: u64,
}

impl VerifiedTpm2Quote {
    pub fn qualification_digest(&self) -> &str {
        &self.qualification_digest
    }
    pub fn report_digest(&self) -> &str {
        &self.report_digest
    }
    pub fn policy_digest(&self) -> &str {
        &self.policy_digest
    }
    pub fn challenge_digest(&self) -> &str {
        &self.challenge_digest
    }
    pub fn ak_binding_digest(&self) -> &str {
        &self.ak_binding_digest
    }
    pub fn quote_artifact_digest(&self) -> &str {
        &self.quote_artifact_digest
    }
    pub fn verification_receipt_digest(&self) -> &str {
        &self.verification_receipt_digest
    }
    pub fn pcr_selection_digest(&self) -> &str {
        &self.pcr_selection_digest
    }
    pub fn pcr_value(&self, pcr: u8) -> Option<[u8; SHA256_LEN]> {
        self.pcr_values
            .iter()
            .find(|value| value.pcr == pcr)
            .map(|value| value.value)
    }
    pub const fn collected_at_ms(&self) -> u64 {
        self.collected_at_ms
    }
    pub const fn verified_at_ms(&self) -> u64 {
        self.verified_at_ms
    }
    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tpm2QuoteQualification {
    pub report: Tpm2QuoteCheckReport,
    pub quote_artifacts: Tpm2QuoteArtifacts,
    pub verification_receipt: QuoteVerificationReceipt,
    verified: VerifiedTpm2Quote,
}

impl Tpm2QuoteQualification {
    pub fn verified(&self) -> &VerifiedTpm2Quote {
        &self.verified
    }
    pub fn into_verified(self) -> VerifiedTpm2Quote {
        self.verified
    }
    pub const fn grants_physical_authority(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct ParsedQuote {
    qualified_signer: Vec<u8>,
    extra_data: Vec<u8>,
    selection: Vec<PcrBankSelection>,
    pcr_digest: [u8; SHA256_LEN],
}

pub fn verify_tpm2_quote(
    policy: &Tpm2CheckquotePolicy,
    challenge: &AttestationChallenge,
    ak: &AttestationKeyBinding,
    bundle: &RawTpm2QuoteBundle,
    verified_at_ms: u64,
    executor: &impl CheckquoteExecutor,
) -> Result<Tpm2QuoteQualification, Tpm2QuoteCheckReport> {
    let policy_digest = policy.canonical_digest();
    let challenge_digest = challenge.validate().then(|| challenge.challenge_digest());
    let ak_binding_digest = ak.validate().then(|| ak.binding_digest());
    let quote_message_digest = digest_bytes(&bundle.quote_message);
    let signature_digest = digest_bytes(&bundle.signature);
    let pcr_values_raw = canonical_pcr_values(&bundle.pcr_values);
    let pcr_values_digest = digest_bytes(&pcr_values_raw);

    let mut report = Tpm2QuoteCheckReport {
        schema_version: CHECKQUOTE_REPORT_SCHEMA_V1.into(),
        adapter_id: policy.adapter_id.clone(),
        policy_digest: policy_digest.clone(),
        challenge_digest: challenge_digest.clone(),
        ak_binding_digest: ak_binding_digest.clone(),
        quote_message_digest: quote_message_digest.clone(),
        signature_digest: signature_digest.clone(),
        pcr_values_digest: pcr_values_digest.clone(),
        parsed_qualified_signer_hex: None,
        parsed_pcr_selection: Vec::new(),
        parsed_pcr_digest_hex: None,
        quote_artifact_digest: None,
        verification_receipt_digest: None,
        verified_at_ms,
        disposition: Tpm2CheckquoteDisposition::Invalid,
        issues: Vec::new(),
    };

    if !policy.validate() {
        report.issues.push(Tpm2CheckquoteIssue::InvalidPolicy);
        return Err(finalize_report(report));
    }
    if !challenge.validate() {
        report.issues.push(Tpm2CheckquoteIssue::InvalidChallenge);
        return Err(finalize_report(report));
    }
    if challenge.verifier_ref != policy.verifier_ref {
        report
            .issues
            .push(Tpm2CheckquoteIssue::ChallengeVerifierMismatch);
        return Err(finalize_report(report));
    }
    if !ak.validate() {
        report.issues.push(Tpm2CheckquoteIssue::InvalidAkBinding);
        return Err(finalize_report(report));
    }
    if ak.name_alg != "sha256" {
        report
            .issues
            .push(Tpm2CheckquoteIssue::UnsupportedAkNameAlgorithm);
        return Err(finalize_report(report));
    }
    if !bundle.validate(policy) {
        report.issues.push(Tpm2CheckquoteIssue::InvalidQuoteBundle);
        return Err(finalize_report(report));
    }
    if bundle.collected_at_ms < challenge.issued_at_ms
        || bundle.collected_at_ms > challenge.expires_at_ms
    {
        report
            .issues
            .push(Tpm2CheckquoteIssue::QuoteOutsideChallengeWindow);
        return Err(finalize_report(report));
    }
    if verified_at_ms < bundle.collected_at_ms {
        report
            .issues
            .push(Tpm2CheckquoteIssue::VerificationBeforeCollection);
        return Err(finalize_report(report));
    }

    let expected_selection = match exact_sha256_selection(challenge) {
        Ok(selection) => selection,
        Err(issue) => {
            report.issues.push(issue);
            return Err(finalize_report(report));
        }
    };
    if bundle.pcr_values.len() != expected_selection.pcrs.len()
        || bundle
            .pcr_values
            .iter()
            .zip(&expected_selection.pcrs)
            .any(|(value, expected)| value.pcr != *expected)
    {
        report
            .issues
            .push(Tpm2CheckquoteIssue::PcrValuesDoNotMatchSelection);
        return Err(finalize_report(report));
    }

    if digest_bytes(&bundle.ak_public) != ak.public_key_digest {
        report
            .issues
            .push(Tpm2CheckquoteIssue::AkPublicDigestMismatch);
        return Err(finalize_report(report));
    }

    let parsed = match parse_tpms_attest_quote(&bundle.quote_message) {
        Ok(value) => value,
        Err(issue) => {
            report.issues.push(issue);
            return Err(finalize_report(report));
        }
    };
    report.parsed_qualified_signer_hex = Some(hex_lower(&parsed.qualified_signer));
    report.parsed_pcr_selection = parsed.selection.clone();
    report.parsed_pcr_digest_hex = Some(hex_lower(&parsed.pcr_digest));

    let expected_signer = match decode_hex(&ak.ak_qualified_name_hex) {
        Some(value) => value,
        None => {
            report.issues.push(Tpm2CheckquoteIssue::InvalidAkBinding);
            return Err(finalize_report(report));
        }
    };
    if parsed.qualified_signer != expected_signer {
        report
            .issues
            .push(Tpm2CheckquoteIssue::QualifiedSignerMismatch);
        return Err(finalize_report(report));
    }
    let expected_nonce = match decode_hex(&challenge.nonce_hex) {
        Some(value) => value,
        None => {
            report.issues.push(Tpm2CheckquoteIssue::InvalidChallenge);
            return Err(finalize_report(report));
        }
    };
    if parsed.extra_data != expected_nonce {
        report
            .issues
            .push(Tpm2CheckquoteIssue::QualifyingDataMismatch);
        return Err(finalize_report(report));
    }
    if parsed.selection.as_slice() != std::slice::from_ref(&expected_selection) {
        report
            .issues
            .push(Tpm2CheckquoteIssue::PcrSelectionMismatch);
        return Err(finalize_report(report));
    }

    let independent_pcr_digest: [u8; SHA256_LEN] = Sha256::digest(&pcr_values_raw).into();
    if parsed.pcr_digest != independent_pcr_digest {
        report
            .issues
            .push(Tpm2CheckquoteIssue::PcrCompositeMismatch);
        return Err(finalize_report(report));
    }

    let actual_tool_digest = match executor.executable_blake3(&policy.checkquote_path) {
        Ok(value) => value,
        Err(error) => {
            report
                .issues
                .push(Tpm2CheckquoteIssue::ToolExecutionUnavailable(error));
            return Err(finalize_report(report));
        }
    };
    if actual_tool_digest != policy.expected_checkquote_blake3 {
        report.issues.push(Tpm2CheckquoteIssue::ToolDigestMismatch);
        return Err(finalize_report(report));
    }

    let request = CheckquoteExecutionRequest {
        ak_public: bundle.ak_public.clone(),
        quote_message: bundle.quote_message.clone(),
        signature: bundle.signature.clone(),
        pcr_values_raw: pcr_values_raw.clone(),
        pcr_list: format!(
            "sha256:{}",
            expected_selection
                .pcrs
                .iter()
                .map(u8::to_string)
                .collect::<Vec<_>>()
                .join(",")
        ),
        qualification_hex: challenge.nonce_hex.to_ascii_lowercase(),
        hash_algorithm: policy.hash_algorithm.clone(),
    };
    let tool_run = match executor.execute_checkquote(&policy.checkquote_path, &request) {
        Ok(value) => value,
        Err(error) => {
            report
                .issues
                .push(Tpm2CheckquoteIssue::ToolExecutionUnavailable(error));
            return Err(finalize_report(report));
        }
    };
    for (stream, bytes) in [("stdout", &tool_run.stdout), ("stderr", &tool_run.stderr)] {
        if bytes.len() as u64 > policy.max_tool_output_bytes {
            report.issues.push(Tpm2CheckquoteIssue::ToolOutputTooLarge {
                stream: stream.into(),
                observed: bytes.len() as u64,
                maximum: policy.max_tool_output_bytes,
            });
            return Err(finalize_report(report));
        }
    }
    if tool_run.exit_code != Some(0) {
        report.issues.push(Tpm2CheckquoteIssue::ToolRejected {
            exit_code: tool_run.exit_code,
        });
        return Err(finalize_report(report));
    }
    if !tool_run.stderr.is_empty() {
        report
            .issues
            .push(Tpm2CheckquoteIssue::ToolEmittedStderr);
        return Err(finalize_report(report));
    }

    let selection_digest = pcr_selection_digest(&challenge.pcr_selection)
        .expect("validated exact SHA-256 selection has a digest");
    let mut evidence_refs = bundle.evidence_refs.clone();
    evidence_refs.extend(policy.evidence_refs.clone());
    evidence_refs.sort();
    evidence_refs.dedup();

    let quote_artifacts = Tpm2QuoteArtifacts {
        schema_version: "1".into(),
        quote_id: bundle.quote_id.clone(),
        challenge_id: challenge.challenge_id.clone(),
        platform_qualification_digest: bundle.platform_qualification_digest.clone(),
        qualifying_data_hex: challenge.nonce_hex.to_ascii_lowercase(),
        quote_message_digest,
        signature_digest,
        pcr_values_digest,
        ak_public_key_digest: ak.public_key_digest.clone(),
        pcr_selection_digest: selection_digest.clone(),
        collected_at_ms: bundle.collected_at_ms,
        evidence_refs: evidence_refs.clone(),
    };
    if !quote_artifacts.validate() {
        report.issues.push(Tpm2CheckquoteIssue::InvalidQuoteBundle);
        return Err(finalize_report(report));
    }
    let quote_artifact_digest = quote_artifacts.artifact_digest();
    report.quote_artifact_digest = Some(quote_artifact_digest.clone());

    let verification_receipt = QuoteVerificationReceipt {
        schema_version: "1".into(),
        receipt_id: format!("{}:{}", policy.adapter_id, bundle.quote_id),
        verifier_ref: policy.verifier_ref.clone(),
        verification_tool_ref: policy.checkquote_path.clone(),
        verification_tool_digest: actual_tool_digest,
        verified_at_ms,
        challenge_digest: challenge.challenge_digest(),
        ak_binding_digest: ak.binding_digest(),
        quote_artifact_digest,
        signature_valid: true,
        qualifying_data_matches: true,
        pcr_digest_matches: true,
        pcr_selection_matches: true,
        attestation_magic_valid: true,
        attestation_type_quote: true,
        evidence_refs,
    };
    if !verification_receipt.validate() {
        report.issues.push(Tpm2CheckquoteIssue::InvalidQuoteBundle);
        return Err(finalize_report(report));
    }
    let receipt_digest = verification_receipt.receipt_digest();
    report.verification_receipt_digest = Some(receipt_digest.clone());
    report.disposition = Tpm2CheckquoteDisposition::Qualified;

    let report_digest = report.canonical_digest();
    let policy_digest = policy_digest.expect("validated policy has digest");
    let challenge_digest = challenge_digest.expect("validated challenge has digest");
    let ak_binding_digest = ak_binding_digest.expect("validated AK has digest");
    let qualification_digest = qualification_digest(
        &report_digest,
        &policy_digest,
        &challenge_digest,
        &ak_binding_digest,
        &quote_artifacts.artifact_digest(),
        &receipt_digest,
        &selection_digest,
        &bundle.pcr_values,
    );
    let verified = VerifiedTpm2Quote {
        qualification_digest,
        report_digest,
        policy_digest,
        challenge_digest,
        ak_binding_digest,
        quote_artifact_digest: quote_artifacts.artifact_digest(),
        verification_receipt_digest: receipt_digest,
        pcr_selection_digest: selection_digest,
        pcr_values: bundle.pcr_values.clone(),
        collected_at_ms: bundle.collected_at_ms,
        verified_at_ms,
    };

    Ok(Tpm2QuoteQualification {
        report,
        quote_artifacts,
        verification_receipt,
        verified,
    })
}

fn parse_tpms_attest_quote(bytes: &[u8]) -> Result<ParsedQuote, Tpm2CheckquoteIssue> {
    let mut cursor = WireCursor::new(bytes);
    let magic = cursor.u32("magic")?;
    if magic != TPM2_GENERATED_VALUE {
        return Err(Tpm2CheckquoteIssue::AttestationMagicMismatch(magic));
    }
    let attest_type = cursor.u16("type")?;
    if attest_type != TPM2_ST_ATTEST_QUOTE {
        return Err(Tpm2CheckquoteIssue::AttestationTypeMismatch(attest_type));
    }
    let qualified_signer = cursor.tpm2b("qualified-signer", MAX_NAME_BYTES)?;
    let extra_data = cursor.tpm2b("extra-data", MAX_EXTRA_DATA_BYTES)?;

    let _clock = cursor.u64("clock")?;
    let _reset_count = cursor.u32("reset-count")?;
    let _restart_count = cursor.u32("restart-count")?;
    let safe = cursor.u8("clock-safe")?;
    if safe > 1 {
        return Err(Tpm2CheckquoteIssue::InvalidClockSafe(safe));
    }
    let _firmware_version = cursor.u64("firmware-version")?;

    let selection_count = cursor.u32("pcr-selection-count")?;
    if selection_count != 1 {
        return Err(Tpm2CheckquoteIssue::UnsupportedPcrSelection);
    }
    let hash_alg = cursor.u16("pcr-hash-algorithm")?;
    if hash_alg != TPM2_ALG_SHA256 {
        return Err(Tpm2CheckquoteIssue::UnsupportedPcrSelection);
    }
    let select_size = cursor.u8("pcr-select-size")? as usize;
    if !(1..=3).contains(&select_size) {
        return Err(Tpm2CheckquoteIssue::UnsupportedPcrSelection);
    }
    let bitmap = cursor.bytes(select_size, "pcr-select")?;
    let mut pcrs = Vec::new();
    for (byte_index, byte) in bitmap.iter().copied().enumerate() {
        for bit in 0..8u8 {
            if byte & (1u8 << bit) != 0 {
                let pcr = byte_index * 8 + bit as usize;
                if pcr > 23 {
                    return Err(Tpm2CheckquoteIssue::UnsupportedPcrSelection);
                }
                pcrs.push(pcr as u8);
            }
        }
    }
    if pcrs.is_empty() {
        return Err(Tpm2CheckquoteIssue::UnsupportedPcrSelection);
    }

    let pcr_digest = cursor.tpm2b("pcr-digest", SHA256_LEN)?;
    if pcr_digest.len() != SHA256_LEN {
        return Err(Tpm2CheckquoteIssue::PcrDigestLengthMismatch(
            pcr_digest.len(),
        ));
    }
    if cursor.remaining() != 0 {
        return Err(Tpm2CheckquoteIssue::QuoteTrailingBytes(cursor.remaining()));
    }

    Ok(ParsedQuote {
        qualified_signer,
        extra_data,
        selection: vec![PcrBankSelection {
            hash_alg: "sha256".into(),
            pcrs,
        }],
        pcr_digest: pcr_digest
            .try_into()
            .expect("SHA-256 digest length was checked"),
    })
}

struct WireCursor<'a> {
    bytes: &'a [u8],
    cursor: usize,
}

impl<'a> WireCursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, cursor: 0 }
    }

    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.cursor)
    }

    fn bytes(&mut self, len: usize, stage: &str) -> Result<Vec<u8>, Tpm2CheckquoteIssue> {
        if self.remaining() < len {
            return Err(Tpm2CheckquoteIssue::QuoteTruncated(stage.into()));
        }
        let start = self.cursor;
        self.cursor += len;
        Ok(self.bytes[start..start + len].to_vec())
    }

    fn u8(&mut self, stage: &str) -> Result<u8, Tpm2CheckquoteIssue> {
        Ok(self.bytes(1, stage)?[0])
    }

    fn u16(&mut self, stage: &str) -> Result<u16, Tpm2CheckquoteIssue> {
        let bytes = self.bytes(2, stage)?;
        Ok(u16::from_be_bytes(bytes.try_into().expect("two bytes")))
    }

    fn u32(&mut self, stage: &str) -> Result<u32, Tpm2CheckquoteIssue> {
        let bytes = self.bytes(4, stage)?;
        Ok(u32::from_be_bytes(bytes.try_into().expect("four bytes")))
    }

    fn u64(&mut self, stage: &str) -> Result<u64, Tpm2CheckquoteIssue> {
        let bytes = self.bytes(8, stage)?;
        Ok(u64::from_be_bytes(bytes.try_into().expect("eight bytes")))
    }

    fn tpm2b(&mut self, stage: &str, maximum: usize) -> Result<Vec<u8>, Tpm2CheckquoteIssue> {
        let len = self.u16(&format!("{stage}-length"))? as usize;
        if len > maximum {
            return Err(Tpm2CheckquoteIssue::QuoteTruncated(format!(
                "{stage}-oversize"
            )));
        }
        self.bytes(len, stage)
    }
}

fn exact_sha256_selection(
    challenge: &AttestationChallenge,
) -> Result<PcrBankSelection, Tpm2CheckquoteIssue> {
    if challenge.pcr_selection.len() != 1 {
        return Err(Tpm2CheckquoteIssue::UnsupportedPcrSelection);
    }
    let selection = challenge.pcr_selection[0].clone();
    if selection.hash_alg != "sha256"
        || selection.pcrs.is_empty()
        || selection.pcrs.len() > 24
        || selection.pcrs.iter().any(|pcr| *pcr > 23)
        || !selection.pcrs.windows(2).all(|pair| pair[0] < pair[1])
    {
        return Err(Tpm2CheckquoteIssue::UnsupportedPcrSelection);
    }
    Ok(selection)
}

fn canonical_pcr_values(values: &[Sha256PcrValue]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len() * SHA256_LEN);
    for value in values {
        bytes.extend_from_slice(&value.value);
    }
    bytes
}

fn qualification_digest(
    report_digest: &str,
    policy_digest: &str,
    challenge_digest: &str,
    ak_binding_digest: &str,
    quote_artifact_digest: &str,
    verification_receipt_digest: &str,
    pcr_selection_digest: &str,
    pcr_values: &[Sha256PcrValue],
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(QUALIFICATION_DIGEST_DOMAIN);
    for value in [
        report_digest,
        policy_digest,
        challenge_digest,
        ak_binding_digest,
        quote_artifact_digest,
        verification_receipt_digest,
        pcr_selection_digest,
    ] {
        push_field(&mut hasher, value);
    }
    for value in pcr_values {
        hasher.update(&[value.pcr]);
        hasher.update(&value.value);
    }
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn finalize_report(mut report: Tpm2QuoteCheckReport) -> Tpm2QuoteCheckReport {
    report.disposition = if report.issues.iter().any(Tpm2CheckquoteIssue::is_invalid) {
        Tpm2CheckquoteDisposition::Invalid
    } else {
        Tpm2CheckquoteDisposition::Blocked
    };
    report
}

fn valid_limit(value: u64, maximum: u64) -> bool {
    value > 0 && value <= maximum
}

fn canonical_text(value: &str) -> bool {
    !value.is_empty()
        && value == value.trim()
        && value.len() <= MAX_TEXT_BYTES
        && !value.chars().any(char::is_control)
}

fn valid_refs(values: &[String]) -> bool {
    !values.is_empty()
        && values.len() <= MAX_EVIDENCE_REFS
        && values.iter().all(|value| canonical_text(value))
        && values.iter().collect::<BTreeSet<_>>().len() == values.len()
}

fn valid_blake3_digest(value: &str) -> bool {
    value.strip_prefix("blake3:").is_some_and(|digest| {
        digest.len() == 64
            && digest.bytes().all(|byte| byte.is_ascii_hexdigit())
            && !digest.bytes().any(|byte| byte.is_ascii_uppercase())
    })
}

fn digest_bytes(bytes: &[u8]) -> String {
    format!("blake3:{}", blake3::hash(bytes).to_hex())
}

fn decode_hex(value: &str) -> Option<Vec<u8>> {
    if value.is_empty() || !value.len().is_multiple_of(2) {
        return None;
    }
    value
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            let text = std::str::from_utf8(pair).ok()?;
            u8::from_str_radix(text, 16).ok()
        })
        .collect()
}

fn hex_lower(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

fn push_field(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_le_bytes());
    hasher.update(value.as_bytes());
}

fn push_sorted_refs(hasher: &mut blake3::Hasher, refs: &[String]) {
    let mut refs = refs.to_vec();
    refs.sort();
    for reference in refs {
        push_field(hasher, &reference);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const PCR10: [u8; 32] = [0x42; 32];
    const NONCE: [u8; 32] = [0xab; 32];
    const QUALIFIED_SIGNER: [u8; 34] = [0x5a; 34];

    #[derive(Clone)]
    struct FakeExecutor {
        digest: String,
        result: Result<ToolExecution, String>,
    }

    impl CheckquoteExecutor for FakeExecutor {
        fn executable_blake3(&self, _executable: &str) -> Result<String, String> {
            Ok(self.digest.clone())
        }

        fn execute_checkquote(
            &self,
            _executable: &str,
            request: &CheckquoteExecutionRequest,
        ) -> Result<ToolExecution, String> {
            assert_eq!(request.pcr_list, "sha256:10");
            assert_eq!(request.qualification_hex, hex_lower(&NONCE));
            assert_eq!(request.pcr_values_raw, PCR10);
            self.result.clone()
        }
    }

    fn digest(bytes: &[u8]) -> String {
        digest_bytes(bytes)
    }

    fn policy(tool_digest: String) -> Tpm2CheckquotePolicy {
        Tpm2CheckquotePolicy {
            schema_version: CHECKQUOTE_POLICY_SCHEMA_V1.into(),
            adapter_id: "adapter:tpm2-checkquote:1".into(),
            verifier_ref: "verifier:remote-1".into(),
            checkquote_path: "/nix/store/example-tpm2-tools/bin/tpm2_checkquote".into(),
            expected_checkquote_blake3: tool_digest,
            hash_algorithm: "sha256".into(),
            max_public_bytes: 1024 * 1024,
            max_message_bytes: 1024 * 1024,
            max_signature_bytes: 1024 * 1024,
            max_tool_output_bytes: 1024 * 1024,
            evidence_refs: vec!["review:tpm2-checkquote-adapter".into()],
        }
    }

    fn challenge() -> AttestationChallenge {
        AttestationChallenge {
            schema_version: "1".into(),
            challenge_id: "challenge:quote:1".into(),
            nonce_hex: hex_lower(&NONCE),
            issued_at_ms: 10_000,
            expires_at_ms: 11_000,
            verifier_ref: "verifier:remote-1".into(),
            pcr_selection: vec![PcrBankSelection {
                hash_alg: "sha256".into(),
                pcrs: vec![10],
            }],
            evidence_refs: vec!["request:quote:1".into()],
        }
    }

    fn ak(public: &[u8]) -> AttestationKeyBinding {
        AttestationKeyBinding {
            schema_version: "1".into(),
            ak_id: "ak:node-1".into(),
            ak_name_hex: "000b01020304".into(),
            ak_qualified_name_hex: hex_lower(&QUALIFIED_SIGNER),
            public_key_digest: digest(public),
            name_alg: "sha256".into(),
            signing_scheme: "ecdsa-sha256".into(),
            fixed_tpm: true,
            restricted_signing: true,
            reviewed_at_ms: 9_500,
            evidence_refs: vec!["review:ak".into()],
        }
    }

    fn quote_message(
        magic: u32,
        attest_type: u16,
        nonce: &[u8],
        signer: &[u8],
        pcrs: &[u8],
        pcr_value: [u8; 32],
    ) -> Vec<u8> {
        let pcr_digest: [u8; 32] = Sha256::digest(pcr_value).into();
        let mut bytes = Vec::new();
        bytes.extend_from_slice(&magic.to_be_bytes());
        bytes.extend_from_slice(&attest_type.to_be_bytes());
        put_tpm2b(&mut bytes, signer);
        put_tpm2b(&mut bytes, nonce);
        bytes.extend_from_slice(&123u64.to_be_bytes());
        bytes.extend_from_slice(&1u32.to_be_bytes());
        bytes.extend_from_slice(&2u32.to_be_bytes());
        bytes.push(1);
        bytes.extend_from_slice(&7u64.to_be_bytes());
        bytes.extend_from_slice(&1u32.to_be_bytes());
        bytes.extend_from_slice(&TPM2_ALG_SHA256.to_be_bytes());
        let mut bitmap = [0u8; 3];
        for pcr in pcrs {
            bitmap[*pcr as usize / 8] |= 1u8 << (*pcr % 8);
        }
        bytes.push(3);
        bytes.extend_from_slice(&bitmap);
        put_tpm2b(&mut bytes, &pcr_digest);
        bytes
    }

    fn put_tpm2b(target: &mut Vec<u8>, value: &[u8]) {
        target.extend_from_slice(&(value.len() as u16).to_be_bytes());
        target.extend_from_slice(value);
    }

    fn bundle(public: &[u8]) -> RawTpm2QuoteBundle {
        RawTpm2QuoteBundle {
            quote_id: "quote:1".into(),
            platform_qualification_digest: digest(b"platform"),
            ak_public: public.to_vec(),
            quote_message: quote_message(
                TPM2_GENERATED_VALUE,
                TPM2_ST_ATTEST_QUOTE,
                &NONCE,
                &QUALIFIED_SIGNER,
                &[10],
                PCR10,
            ),
            signature: b"fake-tss-signature".to_vec(),
            pcr_values: vec![Sha256PcrValue {
                pcr: 10,
                value: PCR10,
            }],
            collected_at_ms: 10_200,
            evidence_refs: vec!["artifact:raw-quote".into()],
        }
    }

    fn executor(tool_digest: String) -> FakeExecutor {
        FakeExecutor {
            digest: tool_digest,
            result: Ok(ToolExecution {
                exit_code: Some(0),
                stdout: Vec::new(),
                stderr: Vec::new(),
            }),
        }
    }

    #[test]
    fn exact_raw_quote_mints_existing_semantic_receipt_and_opaque_capability() {
        let public = b"-----BEGIN PUBLIC KEY-----test-----END PUBLIC KEY-----";
        let tool_digest = digest(b"tpm2-checkquote");
        let result = verify_tpm2_quote(
            &policy(tool_digest.clone()),
            &challenge(),
            &ak(public),
            &bundle(public),
            10_300,
            &executor(tool_digest),
        )
        .unwrap();
        assert_eq!(result.report.disposition, Tpm2CheckquoteDisposition::Qualified);
        assert!(result.verification_receipt.validate());
        assert_eq!(result.verified().pcr_value(10), Some(PCR10));
        assert!(!result.grants_physical_authority());
        assert!(!result.verified().grants_physical_authority());
    }

    #[test]
    fn wrong_magic_is_invalid_independently_of_tool() {
        let public = b"ak-public";
        let tool_digest = digest(b"tool");
        let mut bundle = bundle(public);
        bundle.quote_message[0..4].copy_from_slice(&0u32.to_be_bytes());
        let report = verify_tpm2_quote(
            &policy(tool_digest.clone()),
            &challenge(),
            &ak(public),
            &bundle,
            10_300,
            &executor(tool_digest),
        )
        .unwrap_err();
        assert_eq!(report.disposition, Tpm2CheckquoteDisposition::Invalid);
        assert!(report
            .issues
            .iter()
            .any(|issue| matches!(issue, Tpm2CheckquoteIssue::AttestationMagicMismatch(_))));
    }

    #[test]
    fn wrong_nonce_is_invalid_before_tool_execution() {
        let public = b"ak-public";
        let tool_digest = digest(b"tool");
        let mut bundle = bundle(public);
        bundle.quote_message = quote_message(
            TPM2_GENERATED_VALUE,
            TPM2_ST_ATTEST_QUOTE,
            &[0xcd; 32],
            &QUALIFIED_SIGNER,
            &[10],
            PCR10,
        );
        let report = verify_tpm2_quote(
            &policy(tool_digest.clone()),
            &challenge(),
            &ak(public),
            &bundle,
            10_300,
            &executor(tool_digest),
        )
        .unwrap_err();
        assert!(report
            .issues
            .contains(&Tpm2CheckquoteIssue::QualifyingDataMismatch));
    }

    #[test]
    fn wrong_qualified_signer_is_invalid() {
        let public = b"ak-public";
        let tool_digest = digest(b"tool");
        let mut bundle = bundle(public);
        bundle.quote_message = quote_message(
            TPM2_GENERATED_VALUE,
            TPM2_ST_ATTEST_QUOTE,
            &NONCE,
            &[0x11; 34],
            &[10],
            PCR10,
        );
        let report = verify_tpm2_quote(
            &policy(tool_digest.clone()),
            &challenge(),
            &ak(public),
            &bundle,
            10_300,
            &executor(tool_digest),
        )
        .unwrap_err();
        assert!(report
            .issues
            .contains(&Tpm2CheckquoteIssue::QualifiedSignerMismatch));
    }

    #[test]
    fn quote_selection_substitution_is_invalid() {
        let public = b"ak-public";
        let tool_digest = digest(b"tool");
        let mut bundle = bundle(public);
        bundle.quote_message = quote_message(
            TPM2_GENERATED_VALUE,
            TPM2_ST_ATTEST_QUOTE,
            &NONCE,
            &QUALIFIED_SIGNER,
            &[9],
            PCR10,
        );
        let report = verify_tpm2_quote(
            &policy(tool_digest.clone()),
            &challenge(),
            &ak(public),
            &bundle,
            10_300,
            &executor(tool_digest),
        )
        .unwrap_err();
        assert!(report
            .issues
            .contains(&Tpm2CheckquoteIssue::PcrSelectionMismatch));
    }

    #[test]
    fn pcr_value_substitution_breaks_independent_composite() {
        let public = b"ak-public";
        let tool_digest = digest(b"tool");
        let mut bundle = bundle(public);
        bundle.pcr_values[0].value = [0x99; 32];
        let report = verify_tpm2_quote(
            &policy(tool_digest.clone()),
            &challenge(),
            &ak(public),
            &bundle,
            10_300,
            &executor(tool_digest),
        )
        .unwrap_err();
        assert!(report
            .issues
            .contains(&Tpm2CheckquoteIssue::PcrCompositeMismatch));
    }

    #[test]
    fn success_exit_with_stderr_is_fail_closed() {
        let public = b"ak-public";
        let tool_digest = digest(b"tool");
        let fake = FakeExecutor {
            digest: tool_digest.clone(),
            result: Ok(ToolExecution {
                exit_code: Some(0),
                stdout: Vec::new(),
                stderr: b"unexpected verification diagnostic".to_vec(),
            }),
        };
        let report = verify_tpm2_quote(
            &policy(tool_digest),
            &challenge(),
            &ak(public),
            &bundle(public),
            10_300,
            &fake,
        )
        .unwrap_err();
        assert!(report
            .issues
            .contains(&Tpm2CheckquoteIssue::ToolEmittedStderr));
    }

    #[test]
    fn unreviewed_checkquote_binary_is_blocked_not_reinterpreted_as_bad_quote() {
        let public = b"ak-public";
        let expected = digest(b"expected-tool");
        let fake = executor(digest(b"different-tool"));
        let report = verify_tpm2_quote(
            &policy(expected),
            &challenge(),
            &ak(public),
            &bundle(public),
            10_300,
            &fake,
        )
        .unwrap_err();
        assert_eq!(report.disposition, Tpm2CheckquoteDisposition::Blocked);
        assert!(report
            .issues
            .contains(&Tpm2CheckquoteIssue::ToolDigestMismatch));
    }
}
