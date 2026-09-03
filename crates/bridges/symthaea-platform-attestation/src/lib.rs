// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Fresh TPM2 platform evidence for bounded Symthaea agency.
//!
//! This crate deliberately has no software-qualified fallback. V1 accepts only
//! a locally collected TPM2 quote generated for an exact reviewed PCR selection
//! and independently checked by a reviewed `tpm2_checkquote` binary.
//!
//! TPM `qualifyingData` is intentionally the fixed 32-byte, domain-separated
//! digest of the complete canonical challenge rather than the raw challenge.
//! TPM2B_DATA is hash-sized on real TPMs; hashing preserves binding to the exact
//! capability/workload/configuration/policy/nonce tuple without exceeding that
//! structure's platform-dependent size bound.
//!
//! V1 still trusts the host OS to provide the configured TPM/TCTI and execute
//! the reviewed verifier binaries. IMA/event-log replay and physical-TPM remote
//! attestation can strengthen that root later without changing the opaque proof
//! boundary exposed here.

#![deny(unsafe_code)]

use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use serde::{Deserialize, Serialize};
use symthaea_authority::Digest32;
use thiserror::Error;

pub const PLATFORM_ATTESTATION_SCHEMA_VERSION: u16 = 1;
pub const MAX_APPROVED_PCR_PROFILES: usize = 16;
pub const MAX_SELECTED_PCRS: usize = 24;
pub const MAX_CHALLENGE_AGE_NS: u64 = 60_000_000_000;
pub const MAX_POST_VERIFICATION_AGE_NS: u64 = 60_000_000_000;

const POLICY_DOMAIN: &[u8] = b"symthaea.platform-attestation.policy.v1\0";
const CHALLENGE_DOMAIN: &[u8] = b"symthaea.platform-attestation.challenge.v1\0";
const MAX_TOOL_BYTES: u64 = 64 * 1024 * 1024;
const MAX_AK_PUBLIC_BYTES: u64 = 256 * 1024;
const MAX_QUOTE_MESSAGE_BYTES: u64 = 256 * 1024;
const MAX_QUOTE_SIGNATURE_BYTES: u64 = 256 * 1024;
const MAX_PCR_BLOB_BYTES: u64 = 256 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlatformAttestationPolicyV1 {
    pub schema_version: u16,
    pub policy_id: [u8; 16],
    pub tpm2_quote_path: String,
    pub tpm2_quote_digest: Digest32,
    pub tpm2_checkquote_path: String,
    pub tpm2_checkquote_digest: Digest32,
    pub trusted_ak_public_digest: Digest32,
    /// Exact sorted, unique SHA-256 PCR indices.
    pub sha256_pcr_selection: Vec<u8>,
    /// Sorted, unique commitments to the serialized PCR blob emitted for the
    /// exact selection above.
    pub approved_pcr_profile_digests: Vec<Digest32>,
    pub require_nix_store_tools: bool,
    pub maximum_challenge_age_ns: u64,
    pub maximum_post_verification_age_ns: u64,
}

impl PlatformAttestationPolicyV1 {
    pub fn validate(&self) -> Result<(), PlatformAttestationError> {
        if self.schema_version != PLATFORM_ATTESTATION_SCHEMA_VERSION
            || self.policy_id == [0; 16]
            || self.tpm2_quote_digest.0 == [0; 32]
            || self.tpm2_checkquote_digest.0 == [0; 32]
            || self.trusted_ak_public_digest.0 == [0; 32]
            || !Path::new(&self.tpm2_quote_path).is_absolute()
            || !Path::new(&self.tpm2_checkquote_path).is_absolute()
            || self.sha256_pcr_selection.is_empty()
            || self.sha256_pcr_selection.len() > MAX_SELECTED_PCRS
            || self.approved_pcr_profile_digests.is_empty()
            || self.approved_pcr_profile_digests.len() > MAX_APPROVED_PCR_PROFILES
            || self.maximum_challenge_age_ns == 0
            || self.maximum_challenge_age_ns > MAX_CHALLENGE_AGE_NS
            || self.maximum_post_verification_age_ns == 0
            || self.maximum_post_verification_age_ns > MAX_POST_VERIFICATION_AGE_NS
        {
            return Err(PlatformAttestationError::InvalidPolicy);
        }

        let mut previous_pcr = None;
        for pcr in &self.sha256_pcr_selection {
            if *pcr > 23 || previous_pcr.is_some_and(|old| old >= *pcr) {
                return Err(PlatformAttestationError::InvalidPolicy);
            }
            previous_pcr = Some(*pcr);
        }

        let mut previous_profile = None;
        for profile in &self.approved_pcr_profile_digests {
            if profile.0 == [0; 32] || previous_profile.is_some_and(|old| old >= *profile) {
                return Err(PlatformAttestationError::InvalidPolicy);
            }
            previous_profile = Some(*profile);
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<Digest32, PlatformAttestationError> {
        self.validate()?;
        let mut t = Transcript::new(POLICY_DOMAIN);
        t.u16(self.schema_version);
        t.fixed(&self.policy_id);
        t.bytes(self.tpm2_quote_path.as_bytes())?;
        t.fixed(&self.tpm2_quote_digest.0);
        t.bytes(self.tpm2_checkquote_path.as_bytes())?;
        t.fixed(&self.tpm2_checkquote_digest.0);
        t.fixed(&self.trusted_ak_public_digest.0);
        t.u32(
            u32::try_from(self.sha256_pcr_selection.len())
                .map_err(|_| PlatformAttestationError::Encoding)?,
        );
        for pcr in &self.sha256_pcr_selection {
            t.u8(*pcr);
        }
        t.u32(
            u32::try_from(self.approved_pcr_profile_digests.len())
                .map_err(|_| PlatformAttestationError::Encoding)?,
        );
        for profile in &self.approved_pcr_profile_digests {
            t.fixed(&profile.0);
        }
        t.u8(u8::from(self.require_nix_store_tools));
        t.u64(self.maximum_challenge_age_ns);
        t.u64(self.maximum_post_verification_age_ns);
        Ok(Digest32(t.finish()))
    }

    pub fn pcr_selection_string(&self) -> Result<String, PlatformAttestationError> {
        self.validate()?;
        Ok(format!(
            "sha256:{}",
            self.sha256_pcr_selection
                .iter()
                .map(u8::to_string)
                .collect::<Vec<_>>()
                .join(",")
        ))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct PlatformAttestationChallengeV1 {
    pub schema_version: u16,
    pub nonce: [u8; 32],
    pub grant_digest: Digest32,
    pub workload_digest: Digest32,
    pub configuration_digest: Digest32,
    pub policy_digest: Digest32,
}

impl PlatformAttestationChallengeV1 {
    pub fn canonical_bytes(&self) -> Result<Vec<u8>, PlatformAttestationError> {
        if self.schema_version != PLATFORM_ATTESTATION_SCHEMA_VERSION
            || self.nonce == [0; 32]
            || self.grant_digest.0 == [0; 32]
            || self.workload_digest.0 == [0; 32]
            || self.configuration_digest.0 == [0; 32]
            || self.policy_digest.0 == [0; 32]
        {
            return Err(PlatformAttestationError::InvalidChallenge);
        }
        let mut t = Transcript::new(CHALLENGE_DOMAIN);
        t.u16(self.schema_version);
        t.fixed(&self.nonce);
        t.fixed(&self.grant_digest.0);
        t.fixed(&self.workload_digest.0);
        t.fixed(&self.configuration_digest.0);
        t.fixed(&self.policy_digest.0);
        Ok(t.into_bytes())
    }

    pub fn digest(&self) -> Result<Digest32, PlatformAttestationError> {
        Ok(Digest32(*blake3::hash(&self.canonical_bytes()?).as_bytes()))
    }

    /// Fixed-size TPM qualifying data for this complete challenge.
    pub fn qualification_bytes(&self) -> Result<[u8; 32], PlatformAttestationError> {
        Ok(self.digest()?.0)
    }
}

#[derive(Debug)]
pub struct PendingPlatformAttestationChallenge {
    wire: PlatformAttestationChallengeV1,
    sent_boottime_ns: u64,
}

impl PendingPlatformAttestationChallenge {
    pub fn new(
        policy: &PlatformAttestationPolicyV1,
        grant_digest: Digest32,
        workload_digest: Digest32,
        configuration_digest: Digest32,
    ) -> Result<Self, PlatformAttestationError> {
        policy.validate()?;
        if grant_digest.0 == [0; 32]
            || workload_digest.0 == [0; 32]
            || configuration_digest.0 == [0; 32]
        {
            return Err(PlatformAttestationError::InvalidChallenge);
        }
        let mut nonce = [0u8; 32];
        getrandom::getrandom(&mut nonce)
            .map_err(|_| PlatformAttestationError::RandomnessUnavailable)?;
        if nonce == [0; 32] {
            return Err(PlatformAttestationError::RandomnessUnavailable);
        }
        Ok(Self {
            wire: PlatformAttestationChallengeV1 {
                schema_version: PLATFORM_ATTESTATION_SCHEMA_VERSION,
                nonce,
                grant_digest,
                workload_digest,
                configuration_digest,
                policy_digest: policy.digest()?,
            },
            sent_boottime_ns: linux_boottime_ns()?,
        })
    }

    pub fn wire(&self) -> PlatformAttestationChallengeV1 {
        self.wire
    }
}

#[derive(Debug, Clone)]
pub struct LocalTpm2QuoteInputs {
    /// Context used only to request the quote. Identity terminates in the
    /// separately pinned AK public key checked by `tpm2_checkquote`.
    pub ak_context_path: PathBuf,
    pub ak_public_path: PathBuf,
}

#[derive(Debug)]
pub struct VerifiedPlatformAttestation {
    grant_digest: Digest32,
    workload_digest: Digest32,
    configuration_digest: Digest32,
    policy_digest: Digest32,
    challenge_digest: Digest32,
    ak_public_digest: Digest32,
    pcr_profile_digest: Digest32,
    quote_message_digest: Digest32,
    quote_signature_digest: Digest32,
    quote_tool_digest: Digest32,
    checkquote_tool_digest: Digest32,
    verified_boottime_ns: u64,
    maximum_post_verification_age_ns: u64,
}

impl VerifiedPlatformAttestation {
    pub fn grant_digest(&self) -> Digest32 {
        self.grant_digest
    }
    pub fn workload_digest(&self) -> Digest32 {
        self.workload_digest
    }
    pub fn configuration_digest(&self) -> Digest32 {
        self.configuration_digest
    }
    pub fn policy_digest(&self) -> Digest32 {
        self.policy_digest
    }
    pub fn challenge_digest(&self) -> Digest32 {
        self.challenge_digest
    }
    pub fn ak_public_digest(&self) -> Digest32 {
        self.ak_public_digest
    }
    pub fn pcr_profile_digest(&self) -> Digest32 {
        self.pcr_profile_digest
    }
    pub fn quote_message_digest(&self) -> Digest32 {
        self.quote_message_digest
    }
    pub fn quote_signature_digest(&self) -> Digest32 {
        self.quote_signature_digest
    }

    pub fn ensure_fresh(
        &self,
        policy: &PlatformAttestationPolicyV1,
        grant_digest: Digest32,
        workload_digest: Digest32,
        configuration_digest: Digest32,
    ) -> Result<(), PlatformAttestationError> {
        if policy.digest()? != self.policy_digest {
            return Err(PlatformAttestationError::PolicyMismatch);
        }
        if grant_digest != self.grant_digest
            || workload_digest != self.workload_digest
            || configuration_digest != self.configuration_digest
        {
            return Err(PlatformAttestationError::SubjectMismatch);
        }

        verify_tool_identity(
            Path::new(&policy.tpm2_quote_path),
            policy.tpm2_quote_digest,
            policy.require_nix_store_tools,
        )?;
        verify_tool_identity(
            Path::new(&policy.tpm2_checkquote_path),
            policy.tpm2_checkquote_digest,
            policy.require_nix_store_tools,
        )?;
        if self.quote_tool_digest != policy.tpm2_quote_digest
            || self.checkquote_tool_digest != policy.tpm2_checkquote_digest
        {
            return Err(PlatformAttestationError::ToolIdentityChanged);
        }

        let age = linux_boottime_ns()?
            .checked_sub(self.verified_boottime_ns)
            .ok_or(PlatformAttestationError::BootTimeMovedBackward)?;
        if age > self.maximum_post_verification_age_ns {
            return Err(PlatformAttestationError::VerifiedAttestationStale);
        }
        Ok(())
    }
}

pub fn verify_local_tpm2_attestation_v1(
    policy: &PlatformAttestationPolicyV1,
    challenge: PendingPlatformAttestationChallenge,
    inputs: &LocalTpm2QuoteInputs,
) -> Result<VerifiedPlatformAttestation, PlatformAttestationError> {
    verify_with_runner(policy, challenge, inputs, &SystemTpm2Runner)
}

fn verify_with_runner<R: Tpm2Runner>(
    policy: &PlatformAttestationPolicyV1,
    challenge: PendingPlatformAttestationChallenge,
    inputs: &LocalTpm2QuoteInputs,
    runner: &R,
) -> Result<VerifiedPlatformAttestation, PlatformAttestationError> {
    policy.validate()?;
    if challenge.wire.schema_version != PLATFORM_ATTESTATION_SCHEMA_VERSION
        || challenge.wire.policy_digest != policy.digest()?
    {
        return Err(PlatformAttestationError::InvalidChallenge);
    }

    let received_boottime_ns = linux_boottime_ns()?;
    let challenge_age_ns = received_boottime_ns
        .checked_sub(challenge.sent_boottime_ns)
        .ok_or(PlatformAttestationError::BootTimeMovedBackward)?;
    if challenge_age_ns > policy.maximum_challenge_age_ns {
        return Err(PlatformAttestationError::ChallengeExpired);
    }

    let quote_tool = verify_tool_identity(
        Path::new(&policy.tpm2_quote_path),
        policy.tpm2_quote_digest,
        policy.require_nix_store_tools,
    )?;
    let checkquote_tool = verify_tool_identity(
        Path::new(&policy.tpm2_checkquote_path),
        policy.tpm2_checkquote_digest,
        policy.require_nix_store_tools,
    )?;

    let ak_public = read_bounded(&inputs.ak_public_path, MAX_AK_PUBLIC_BYTES)?;
    let ak_public_digest = digest_bytes(&ak_public);
    if ak_public_digest != policy.trusted_ak_public_digest {
        return Err(PlatformAttestationError::AkPublicKeyMismatch);
    }

    let temp = EvidenceDir::new()?;
    let qualification_path = temp.path.join("qualification.bin");
    let message_path = temp.path.join("quote.msg");
    let signature_path = temp.path.join("quote.sig");
    let pcrs_path = temp.path.join("quote.pcrs");
    let ak_public_copy_path = temp.path.join("ak-public.bin");

    let challenge_digest = challenge.wire.digest()?;
    fs::write(&qualification_path, challenge.wire.qualification_bytes()?)?;
    fs::write(&ak_public_copy_path, &ak_public)?;

    let selection = policy.pcr_selection_string()?;
    runner.quote(QuoteInvocation {
        tool: &quote_tool,
        ak_context: &inputs.ak_context_path,
        selection: &selection,
        qualification: &qualification_path,
        message: &message_path,
        signature: &signature_path,
        pcrs: &pcrs_path,
    })?;

    let pcr_blob = read_bounded(&pcrs_path, MAX_PCR_BLOB_BYTES)?;
    let pcr_profile_digest = digest_bytes(&pcr_blob);
    if policy
        .approved_pcr_profile_digests
        .binary_search(&pcr_profile_digest)
        .is_err()
    {
        return Err(PlatformAttestationError::PcrProfileNotApproved);
    }

    runner.checkquote(CheckQuoteInvocation {
        tool: &checkquote_tool,
        ak_public: &ak_public_copy_path,
        qualification: &qualification_path,
        message: &message_path,
        signature: &signature_path,
        pcrs: &pcrs_path,
    })?;

    let quote_message = read_bounded(&message_path, MAX_QUOTE_MESSAGE_BYTES)?;
    let quote_signature = read_bounded(&signature_path, MAX_QUOTE_SIGNATURE_BYTES)?;
    if quote_message.is_empty() || quote_signature.is_empty() || pcr_blob.is_empty() {
        return Err(PlatformAttestationError::EmptyQuoteEvidence);
    }

    Ok(VerifiedPlatformAttestation {
        grant_digest: challenge.wire.grant_digest,
        workload_digest: challenge.wire.workload_digest,
        configuration_digest: challenge.wire.configuration_digest,
        policy_digest: challenge.wire.policy_digest,
        challenge_digest,
        ak_public_digest,
        pcr_profile_digest,
        quote_message_digest: digest_bytes(&quote_message),
        quote_signature_digest: digest_bytes(&quote_signature),
        quote_tool_digest: policy.tpm2_quote_digest,
        checkquote_tool_digest: policy.tpm2_checkquote_digest,
        verified_boottime_ns: linux_boottime_ns()?,
        maximum_post_verification_age_ns: policy.maximum_post_verification_age_ns,
    })
}

struct QuoteInvocation<'a> {
    tool: &'a Path,
    ak_context: &'a Path,
    selection: &'a str,
    qualification: &'a Path,
    message: &'a Path,
    signature: &'a Path,
    pcrs: &'a Path,
}

struct CheckQuoteInvocation<'a> {
    tool: &'a Path,
    ak_public: &'a Path,
    qualification: &'a Path,
    message: &'a Path,
    signature: &'a Path,
    pcrs: &'a Path,
}

trait Tpm2Runner {
    fn quote(&self, invocation: QuoteInvocation<'_>) -> Result<(), PlatformAttestationError>;
    fn checkquote(
        &self,
        invocation: CheckQuoteInvocation<'_>,
    ) -> Result<(), PlatformAttestationError>;
}

struct SystemTpm2Runner;

impl Tpm2Runner for SystemTpm2Runner {
    fn quote(&self, invocation: QuoteInvocation<'_>) -> Result<(), PlatformAttestationError> {
        let output = Command::new(invocation.tool)
            .arg("-Q")
            .arg("-c")
            .arg(invocation.ak_context)
            .arg("-l")
            .arg(invocation.selection)
            .arg("-q")
            .arg(invocation.qualification)
            .arg("-m")
            .arg(invocation.message)
            .arg("-s")
            .arg(invocation.signature)
            .arg("-o")
            .arg(invocation.pcrs)
            .arg("-g")
            .arg("sha256")
            .output()?;
        if !output.status.success() {
            return Err(PlatformAttestationError::QuoteToolFailed(digest_bytes(
                &output.stderr,
            )));
        }
        Ok(())
    }

    fn checkquote(
        &self,
        invocation: CheckQuoteInvocation<'_>,
    ) -> Result<(), PlatformAttestationError> {
        let output = Command::new(invocation.tool)
            .arg("-u")
            .arg(invocation.ak_public)
            .arg("-m")
            .arg(invocation.message)
            .arg("-s")
            .arg(invocation.signature)
            .arg("-f")
            .arg(invocation.pcrs)
            .arg("-g")
            .arg("sha256")
            .arg("-q")
            .arg(invocation.qualification)
            .output()?;
        if !output.status.success() {
            return Err(PlatformAttestationError::CheckQuoteFailed(digest_bytes(
                &output.stderr,
            )));
        }
        Ok(())
    }
}

fn verify_tool_identity(
    configured_path: &Path,
    expected_digest: Digest32,
    require_nix_store: bool,
) -> Result<PathBuf, PlatformAttestationError> {
    let resolved = fs::canonicalize(configured_path)?;
    if require_nix_store && !path_is_in_nix_store(&resolved) {
        return Err(PlatformAttestationError::ToolOutsideNixStore);
    }
    if digest_bytes(&read_bounded(&resolved, MAX_TOOL_BYTES)?) != expected_digest {
        return Err(PlatformAttestationError::ToolDigestMismatch);
    }
    Ok(resolved)
}

fn path_is_in_nix_store(path: &Path) -> bool {
    path.components().take(3).collect::<PathBuf>() == PathBuf::from("/nix/store")
}

fn read_bounded(path: &Path, maximum_bytes: u64) -> Result<Vec<u8>, PlatformAttestationError> {
    let metadata = fs::metadata(path)?;
    if !metadata.is_file() || metadata.len() == 0 || metadata.len() > maximum_bytes {
        return Err(PlatformAttestationError::InvalidEvidenceFile);
    }
    let bytes = fs::read(path)?;
    if u64::try_from(bytes.len()).map_err(|_| PlatformAttestationError::Encoding)? > maximum_bytes {
        return Err(PlatformAttestationError::InvalidEvidenceFile);
    }
    Ok(bytes)
}

fn digest_bytes(bytes: &[u8]) -> Digest32 {
    Digest32(*blake3::hash(bytes).as_bytes())
}

struct EvidenceDir {
    path: PathBuf,
}

impl EvidenceDir {
    fn new() -> Result<Self, PlatformAttestationError> {
        let mut nonce = [0u8; 16];
        getrandom::getrandom(&mut nonce)
            .map_err(|_| PlatformAttestationError::RandomnessUnavailable)?;
        let suffix = nonce.iter().map(|b| format!("{b:02x}")).collect::<String>();
        let path = std::env::temp_dir().join(format!("symthaea-tpm2-{suffix}"));
        fs::create_dir(&path)?;
        Ok(Self { path })
    }
}

impl Drop for EvidenceDir {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.path);
    }
}

fn linux_boottime_ns() -> Result<u64, PlatformAttestationError> {
    let uptime = fs::read_to_string("/proc/uptime")?;
    decimal_seconds_to_ns(
        uptime
            .split_whitespace()
            .next()
            .ok_or(PlatformAttestationError::InvalidBootTime)?,
    )
}

fn decimal_seconds_to_ns(value: &str) -> Result<u64, PlatformAttestationError> {
    let (seconds, fractional) = value.split_once('.').unwrap_or((value, ""));
    let seconds = seconds
        .parse::<u64>()
        .map_err(|_| PlatformAttestationError::InvalidBootTime)?;
    if fractional.len() > 9 || !fractional.bytes().all(|b| b.is_ascii_digit()) {
        return Err(PlatformAttestationError::InvalidBootTime);
    }
    let mut nanos = fractional.to_string();
    while nanos.len() < 9 {
        nanos.push('0');
    }
    let nanos = nanos
        .parse::<u64>()
        .map_err(|_| PlatformAttestationError::InvalidBootTime)?;
    seconds
        .checked_mul(1_000_000_000)
        .and_then(|base| base.checked_add(nanos))
        .ok_or(PlatformAttestationError::ArithmeticOverflow)
}

#[derive(Debug, Error)]
pub enum PlatformAttestationError {
    #[error("invalid platform-attestation policy")]
    InvalidPolicy,
    #[error("invalid platform-attestation challenge")]
    InvalidChallenge,
    #[error("platform-attestation challenge expired")]
    ChallengeExpired,
    #[error("verified platform attestation expired")]
    VerifiedAttestationStale,
    #[error("platform-attestation policy changed")]
    PolicyMismatch,
    #[error("platform-attestation subject changed")]
    SubjectMismatch,
    #[error("TPM2 verifier tool resolved outside required Nix store")]
    ToolOutsideNixStore,
    #[error("TPM2 verifier tool digest does not match reviewed policy")]
    ToolDigestMismatch,
    #[error("TPM2 verifier tool identity changed after verification")]
    ToolIdentityChanged,
    #[error("attestation-key public bytes do not match reviewed policy")]
    AkPublicKeyMismatch,
    #[error("fresh TPM quote PCR state is not an approved profile")]
    PcrProfileNotApproved,
    #[error("tpm2_quote failed; stderr commitment {0:?}")]
    QuoteToolFailed(Digest32),
    #[error("tpm2_checkquote rejected quote/signature/nonce/PCR evidence; stderr commitment {0:?}")]
    CheckQuoteFailed(Digest32),
    #[error("quote evidence output was empty")]
    EmptyQuoteEvidence,
    #[error("invalid or oversized attestation evidence file")]
    InvalidEvidenceFile,
    #[error("OS randomness unavailable")]
    RandomnessUnavailable,
    #[error("Linux boot-time source is malformed")]
    InvalidBootTime,
    #[error("Linux boot time moved backwards")]
    BootTimeMovedBackward,
    #[error("arithmetic overflow")]
    ArithmeticOverflow,
    #[error("canonical encoding failed")]
    Encoding,
    #[error("I/O failure: {0}")]
    Io(#[from] std::io::Error),
}

struct Transcript {
    bytes: Vec<u8>,
}

impl Transcript {
    fn new(domain: &[u8]) -> Self {
        let mut bytes = Vec::with_capacity(domain.len() + 256);
        bytes.extend_from_slice(domain);
        Self { bytes }
    }
    fn u8(&mut self, value: u8) {
        self.bytes.push(value);
    }
    fn u16(&mut self, value: u16) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }
    fn u32(&mut self, value: u32) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }
    fn u64(&mut self, value: u64) {
        self.bytes.extend_from_slice(&value.to_be_bytes());
    }
    fn fixed(&mut self, value: &[u8]) {
        self.bytes.extend_from_slice(value);
    }
    fn bytes(&mut self, value: &[u8]) -> Result<(), PlatformAttestationError> {
        self.u32(u32::try_from(value.len()).map_err(|_| PlatformAttestationError::Encoding)?);
        self.bytes.extend_from_slice(value);
        Ok(())
    }
    fn finish(self) -> [u8; 32] {
        *blake3::hash(&self.bytes).as_bytes()
    }
    fn into_bytes(self) -> Vec<u8> {
        self.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    fn temp_file(name: &str, bytes: &[u8]) -> PathBuf {
        let mut nonce = [0u8; 8];
        getrandom::getrandom(&mut nonce).unwrap();
        let suffix = nonce.iter().map(|b| format!("{b:02x}")).collect::<String>();
        let path = std::env::temp_dir().join(format!("symthaea-platform-{name}-{suffix}"));
        fs::write(&path, bytes).unwrap();
        path
    }

    fn policy(
        quote_tool: &Path,
        check_tool: &Path,
        ak_public: &Path,
        pcr: Digest32,
    ) -> PlatformAttestationPolicyV1 {
        PlatformAttestationPolicyV1 {
            schema_version: PLATFORM_ATTESTATION_SCHEMA_VERSION,
            policy_id: [1; 16],
            tpm2_quote_path: quote_tool.to_string_lossy().into_owned(),
            tpm2_quote_digest: digest_bytes(&fs::read(quote_tool).unwrap()),
            tpm2_checkquote_path: check_tool.to_string_lossy().into_owned(),
            tpm2_checkquote_digest: digest_bytes(&fs::read(check_tool).unwrap()),
            trusted_ak_public_digest: digest_bytes(&fs::read(ak_public).unwrap()),
            sha256_pcr_selection: vec![0, 7, 16],
            approved_pcr_profile_digests: vec![pcr],
            require_nix_store_tools: false,
            maximum_challenge_age_ns: 5_000_000_000,
            maximum_post_verification_age_ns: 5_000_000_000,
        }
    }

    struct FakeRunner {
        pcr_blob: Vec<u8>,
        expected_qualification: [u8; 32],
        check_calls: Mutex<u32>,
    }

    impl Tpm2Runner for FakeRunner {
        fn quote(&self, invocation: QuoteInvocation<'_>) -> Result<(), PlatformAttestationError> {
            assert_eq!(invocation.selection, "sha256:0,7,16");
            let qualification = fs::read(invocation.qualification)?;
            assert_eq!(qualification.len(), 32);
            assert_eq!(qualification, self.expected_qualification);
            fs::write(invocation.message, b"signed quote message")?;
            fs::write(invocation.signature, b"tpm signature")?;
            fs::write(invocation.pcrs, &self.pcr_blob)?;
            Ok(())
        }

        fn checkquote(
            &self,
            invocation: CheckQuoteInvocation<'_>,
        ) -> Result<(), PlatformAttestationError> {
            assert_eq!(fs::read(invocation.qualification)?, self.expected_qualification);
            assert!(!fs::read(invocation.ak_public)?.is_empty());
            assert_eq!(fs::read(invocation.message)?, b"signed quote message");
            assert_eq!(fs::read(invocation.signature)?, b"tpm signature");
            assert_eq!(fs::read(invocation.pcrs)?, self.pcr_blob);
            *self.check_calls.lock().unwrap() += 1;
            Ok(())
        }
    }

    #[test]
    fn challenge_qualification_is_exactly_32_byte_digest() {
        let tool_a = temp_file("tool-a", b"quote-tool");
        let tool_b = temp_file("tool-b", b"check-tool");
        let ak = temp_file("ak", b"ak-public");
        let p = policy(&tool_a, &tool_b, &ak, digest_bytes(b"pcrs"));
        let pending = PendingPlatformAttestationChallenge::new(
            &p,
            Digest32([2; 32]),
            Digest32([3; 32]),
            Digest32([4; 32]),
        )
        .unwrap();
        let wire = pending.wire();
        assert_eq!(wire.qualification_bytes().unwrap().len(), 32);
        assert_eq!(wire.qualification_bytes().unwrap(), wire.digest().unwrap().0);
        let _ = fs::remove_file(tool_a);
        let _ = fs::remove_file(tool_b);
        let _ = fs::remove_file(ak);
    }

    #[test]
    fn exact_pcr_selection_is_canonical() {
        let tool_a = temp_file("tool-a", b"quote-tool");
        let tool_b = temp_file("tool-b", b"check-tool");
        let ak = temp_file("ak", b"ak-public");
        let p = policy(&tool_a, &tool_b, &ak, digest_bytes(b"pcrs"));
        assert_eq!(p.pcr_selection_string().unwrap(), "sha256:0,7,16");
        let _ = fs::remove_file(tool_a);
        let _ = fs::remove_file(tool_b);
        let _ = fs::remove_file(ak);
    }

    #[test]
    fn unsorted_or_duplicate_pcrs_fail_policy_validation() {
        let tool_a = temp_file("tool-a", b"quote-tool");
        let tool_b = temp_file("tool-b", b"check-tool");
        let ak = temp_file("ak", b"ak-public");
        let mut p = policy(&tool_a, &tool_b, &ak, digest_bytes(b"pcrs"));
        p.sha256_pcr_selection = vec![7, 0];
        assert!(matches!(p.validate(), Err(PlatformAttestationError::InvalidPolicy)));
        p.sha256_pcr_selection = vec![7, 7];
        assert!(matches!(p.validate(), Err(PlatformAttestationError::InvalidPolicy)));
        let _ = fs::remove_file(tool_a);
        let _ = fs::remove_file(tool_b);
        let _ = fs::remove_file(ak);
    }

    #[test]
    fn fake_runner_can_only_produce_proof_for_approved_profile() {
        let tool_a = temp_file("tool-a", b"quote-tool");
        let tool_b = temp_file("tool-b", b"check-tool");
        let ak = temp_file("ak", b"ak-public");
        let pcr_blob = b"serialized-pcr-profile".to_vec();
        let p = policy(&tool_a, &tool_b, &ak, digest_bytes(&pcr_blob));
        let challenge = PendingPlatformAttestationChallenge::new(
            &p,
            Digest32([2; 32]),
            Digest32([3; 32]),
            Digest32([4; 32]),
        )
        .unwrap();
        let expected_qualification = challenge.wire().qualification_bytes().unwrap();
        let runner = FakeRunner {
            pcr_blob: pcr_blob.clone(),
            expected_qualification,
            check_calls: Mutex::new(0),
        };
        let verified = verify_with_runner(
            &p,
            challenge,
            &LocalTpm2QuoteInputs {
                ak_context_path: PathBuf::from("unused-by-fake"),
                ak_public_path: ak.clone(),
            },
            &runner,
        )
        .unwrap();
        assert_eq!(verified.pcr_profile_digest(), digest_bytes(&pcr_blob));
        assert_eq!(*runner.check_calls.lock().unwrap(), 1);
        let _ = fs::remove_file(tool_a);
        let _ = fs::remove_file(tool_b);
        let _ = fs::remove_file(ak);
    }

    #[test]
    fn unapproved_pcr_profile_fails_before_checkquote() {
        let tool_a = temp_file("tool-a", b"quote-tool");
        let tool_b = temp_file("tool-b", b"check-tool");
        let ak = temp_file("ak", b"ak-public");
        let p = policy(&tool_a, &tool_b, &ak, digest_bytes(b"approved"));
        let challenge = PendingPlatformAttestationChallenge::new(
            &p,
            Digest32([2; 32]),
            Digest32([3; 32]),
            Digest32([4; 32]),
        )
        .unwrap();
        let expected_qualification = challenge.wire().qualification_bytes().unwrap();
        let runner = FakeRunner {
            pcr_blob: b"different".to_vec(),
            expected_qualification,
            check_calls: Mutex::new(0),
        };
        assert!(matches!(
            verify_with_runner(
                &p,
                challenge,
                &LocalTpm2QuoteInputs {
                    ak_context_path: PathBuf::from("unused-by-fake"),
                    ak_public_path: ak.clone(),
                },
                &runner,
            ),
            Err(PlatformAttestationError::PcrProfileNotApproved)
        ));
        assert_eq!(*runner.check_calls.lock().unwrap(), 0);
        let _ = fs::remove_file(tool_a);
        let _ = fs::remove_file(tool_b);
        let _ = fs::remove_file(ak);
    }

    #[test]
    fn policy_digest_changes_with_verifier_tool_or_pcr_profile() {
        let tool_a = temp_file("tool-a", b"quote-tool");
        let tool_b = temp_file("tool-b", b"check-tool");
        let ak = temp_file("ak", b"ak-public");
        let p = policy(&tool_a, &tool_b, &ak, digest_bytes(b"pcr-a"));
        let d1 = p.digest().unwrap();
        let mut changed = p.clone();
        changed.approved_pcr_profile_digests = vec![digest_bytes(b"pcr-b")];
        assert_ne!(d1, changed.digest().unwrap());
        let mut changed = p.clone();
        changed.tpm2_checkquote_digest = Digest32([9; 32]);
        assert_ne!(d1, changed.digest().unwrap());
        let _ = fs::remove_file(tool_a);
        let _ = fs::remove_file(tool_b);
        let _ = fs::remove_file(ak);
    }

    #[test]
    fn decimal_uptime_parser_avoids_float_rounding() {
        assert_eq!(decimal_seconds_to_ns("12.34").unwrap(), 12_340_000_000);
        assert_eq!(decimal_seconds_to_ns("1.000000001").unwrap(), 1_000_000_001);
        assert!(decimal_seconds_to_ns("1.0000000001").is_err());
    }
}
