// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Independent witness attestations for accepted Agency qualification evidence.
//!
//! This crate does not decide whether a qualification passed. It consumes the
//! closed-world acceptance JSON emitted by `verify-tpm2-qualification-evidence.py`
//! and only accepts the stronger release-bound form: archive SHA-256, Git HEAD,
//! and Git tree must already have been supplied to that verifier independently.
//!
//! A witness signs a domain-separated commitment to the interpreted acceptance,
//! the exact evidence-verifier implementation digest, and a reviewed witness
//! policy. A quorum can require both witness count and organization/service
//! diversity. The result is evidence/notarization only; it carries no execution
//! authority and cannot make an invalid qualification valid.

#![deny(unsafe_code)]

use std::collections::BTreeSet;

use ed25519_dalek::{Signature, Signer, SigningKey, VerifyingKey};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use symthaea_authority::Digest32;
use thiserror::Error;

pub const ACCEPTANCE_SCHEMA: &str = "symthaea.agency-tpm2-evidence-acceptance.v1";
pub const WITNESS_SCHEMA_VERSION: u16 = 1;
pub const SIGNATURE_ALGORITHM: &str = "ed25519-rfc8032";
pub const MAX_ACCEPTANCE_JSON_BYTES: usize = 128 * 1024;
pub const MAX_NIXPKGS_LOCKED_CANONICAL_BYTES: usize = 64 * 1024;
pub const MAX_VERIFIER_STORE_BYTES: usize = 4096;
pub const MAX_WITNESSES: usize = 64;
pub const MAX_ALLOWED_VERIFIERS: usize = 32;

const ACCEPTANCE_DOMAIN: &[u8] = b"symthaea.qualification-acceptance.v1\0";
const NIXPKGS_DOMAIN: &[u8] = b"symthaea.qualification-acceptance.nixpkgs-locked.v1\0";
const POLICY_DOMAIN: &[u8] = b"symthaea.qualification-witness.policy.v1\0";
const ATTESTATION_DOMAIN: &[u8] = b"symthaea.qualification-witness.attestation.v1\0";

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawAcceptanceV1 {
    schema: String,
    accepted: bool,
    qualification_result: String,
    archive_sha256: String,
    archive_hash_source: String,
    manifest_sha256: String,
    head: String,
    tree: String,
    external_head_bound: bool,
    external_tree_bound: bool,
    release_bound: bool,
    nixpkgs_locked: Value,
    flake_lock_sha256: String,
    rust_toolchain_sha256: String,
    approved_pcr_profile: String,
    policy_digest: String,
    ak_public_digest: String,
    challenge_digest: String,
    probe_sha256: String,
    quote_wrapper_sha256: String,
    checkquote_wrapper_sha256: String,
    verifier_store: String,
}

/// Opaque interpretation of a complete release-bound V1 evidence acceptance.
#[derive(Debug)]
pub struct QualificationAcceptanceV1 {
    archive_sha256: Digest32,
    manifest_sha256: Digest32,
    head: [u8; 20],
    tree: [u8; 20],
    nixpkgs_locked_digest: Digest32,
    flake_lock_sha256: Digest32,
    rust_toolchain_sha256: Digest32,
    approved_pcr_profile: Digest32,
    platform_policy_digest: Digest32,
    ak_public_digest: Digest32,
    challenge_digest: Digest32,
    probe_sha256: Digest32,
    quote_wrapper_sha256: Digest32,
    checkquote_wrapper_sha256: Digest32,
    verifier_store: String,
}

impl QualificationAcceptanceV1 {
    pub fn archive_sha256(&self) -> Digest32 {
        self.archive_sha256
    }

    pub fn head(&self) -> [u8; 20] {
        self.head
    }

    pub fn tree(&self) -> [u8; 20] {
        self.tree
    }

    pub fn approved_pcr_profile(&self) -> Digest32 {
        self.approved_pcr_profile
    }

    pub fn digest(&self) -> Result<Digest32, QualificationWitnessError> {
        let mut transcript = Transcript::new(ACCEPTANCE_DOMAIN);
        transcript.u16(WITNESS_SCHEMA_VERSION);
        transcript.fixed(&self.archive_sha256.0);
        transcript.fixed(&self.manifest_sha256.0);
        transcript.fixed(&self.head);
        transcript.fixed(&self.tree);
        transcript.fixed(&self.nixpkgs_locked_digest.0);
        transcript.fixed(&self.flake_lock_sha256.0);
        transcript.fixed(&self.rust_toolchain_sha256.0);
        transcript.fixed(&self.approved_pcr_profile.0);
        transcript.fixed(&self.platform_policy_digest.0);
        transcript.fixed(&self.ak_public_digest.0);
        transcript.fixed(&self.challenge_digest.0);
        transcript.fixed(&self.probe_sha256.0);
        transcript.fixed(&self.quote_wrapper_sha256.0);
        transcript.fixed(&self.checkquote_wrapper_sha256.0);
        transcript.bytes(self.verifier_store.as_bytes())?;
        Ok(Digest32(transcript.finish()))
    }
}

/// Parse only the release form of #431's acceptance statement.
///
/// Sidecar-only or otherwise unanchored acceptances are intentionally rejected;
/// a later witness must not be able to upgrade weak evidence into release proof.
pub fn parse_release_acceptance_v1(
    json_bytes: &[u8],
) -> Result<QualificationAcceptanceV1, QualificationWitnessError> {
    if json_bytes.is_empty() || json_bytes.len() > MAX_ACCEPTANCE_JSON_BYTES {
        return Err(QualificationWitnessError::InvalidAcceptance);
    }
    let raw: RawAcceptanceV1 = serde_json::from_slice(json_bytes)
        .map_err(|_| QualificationWitnessError::InvalidAcceptance)?;
    if raw.schema != ACCEPTANCE_SCHEMA
        || !raw.accepted
        || raw.qualification_result != "PASS"
        || raw.archive_hash_source != "caller"
        || !raw.external_head_bound
        || !raw.external_tree_bound
        || !raw.release_bound
    {
        return Err(QualificationWitnessError::AcceptanceNotReleaseBound);
    }
    if raw.verifier_store.len() > MAX_VERIFIER_STORE_BYTES
        || !raw.verifier_store.starts_with("/nix/store/")
        || raw
            .verifier_store
            .bytes()
            .any(|byte| byte == 0 || byte.is_ascii_control())
    {
        return Err(QualificationWitnessError::InvalidAcceptance);
    }

    let canonical_nixpkgs = canonical_json_bytes(&raw.nixpkgs_locked)?;
    if canonical_nixpkgs.len() > MAX_NIXPKGS_LOCKED_CANONICAL_BYTES {
        return Err(QualificationWitnessError::InvalidAcceptance);
    }
    let mut nixpkgs_hasher = blake3::Hasher::new();
    nixpkgs_hasher.update(NIXPKGS_DOMAIN);
    nixpkgs_hasher.update(&canonical_nixpkgs);

    Ok(QualificationAcceptanceV1 {
        archive_sha256: parse_digest32(&raw.archive_sha256)?,
        manifest_sha256: parse_digest32(&raw.manifest_sha256)?,
        head: parse_hex20(&raw.head)?,
        tree: parse_hex20(&raw.tree)?,
        nixpkgs_locked_digest: Digest32(*nixpkgs_hasher.finalize().as_bytes()),
        flake_lock_sha256: parse_digest32(&raw.flake_lock_sha256)?,
        rust_toolchain_sha256: parse_digest32(&raw.rust_toolchain_sha256)?,
        approved_pcr_profile: parse_digest32(&raw.approved_pcr_profile)?,
        platform_policy_digest: parse_digest32(&raw.policy_digest)?,
        ak_public_digest: parse_digest32(&raw.ak_public_digest)?,
        challenge_digest: parse_digest32(&raw.challenge_digest)?,
        probe_sha256: parse_digest32(&raw.probe_sha256)?,
        quote_wrapper_sha256: parse_digest32(&raw.quote_wrapper_sha256)?,
        checkquote_wrapper_sha256: parse_digest32(&raw.checkquote_wrapper_sha256)?,
        verifier_store: raw.verifier_store,
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QualificationWitnessIdentityV1 {
    pub witness_id: [u8; 16],
    pub organization_id: [u8; 16],
    pub service_id: [u8; 16],
    pub public_key: [u8; 32],
}

impl QualificationWitnessIdentityV1 {
    fn validate(&self) -> Result<(), QualificationWitnessError> {
        if self.witness_id == [0; 16]
            || self.organization_id == [0; 16]
            || self.service_id == [0; 16]
            || self.public_key == [0; 32]
        {
            return Err(QualificationWitnessError::InvalidWitnessIdentity);
        }
        VerifyingKey::from_bytes(&self.public_key)
            .map_err(|_| QualificationWitnessError::InvalidWitnessIdentity)?;
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QualificationWitnessPolicyV1 {
    pub schema_version: u16,
    pub policy_id: [u8; 16],
    /// Rotating this epoch invalidates witness attestations from an older
    /// key/policy generation without changing the underlying acceptance.
    pub witness_epoch: u64,
    pub threshold: u16,
    pub minimum_organizations: u16,
    pub minimum_services: u16,
    /// Exact accepted SHA/BLAKE3 commitments of evidence-verifier implementations.
    pub allowed_verifier_digests: Vec<Digest32>,
    /// Canonically ordered by witness_id.
    pub witnesses: Vec<QualificationWitnessIdentityV1>,
}

impl QualificationWitnessPolicyV1 {
    pub fn validate(&self) -> Result<(), QualificationWitnessError> {
        if self.schema_version != WITNESS_SCHEMA_VERSION
            || self.policy_id == [0; 16]
            || self.witness_epoch == 0
            || self.threshold == 0
            || self.minimum_organizations == 0
            || self.minimum_services == 0
            || self.witnesses.is_empty()
            || self.witnesses.len() > MAX_WITNESSES
            || self.allowed_verifier_digests.is_empty()
            || self.allowed_verifier_digests.len() > MAX_ALLOWED_VERIFIERS
            || usize::from(self.threshold) > self.witnesses.len()
            || self.minimum_organizations > self.threshold
            || self.minimum_services > self.threshold
        {
            return Err(QualificationWitnessError::InvalidWitnessPolicy);
        }

        let mut previous_witness = None;
        let mut organizations = BTreeSet::new();
        let mut services = BTreeSet::new();
        for witness in &self.witnesses {
            witness.validate()?;
            if previous_witness.is_some_and(|old| old >= witness.witness_id) {
                return Err(QualificationWitnessError::InvalidWitnessPolicy);
            }
            previous_witness = Some(witness.witness_id);
            organizations.insert(witness.organization_id);
            services.insert(witness.service_id);
        }
        if organizations.len() < usize::from(self.minimum_organizations)
            || services.len() < usize::from(self.minimum_services)
        {
            return Err(QualificationWitnessError::InvalidWitnessPolicy);
        }

        let mut previous_verifier = None;
        for verifier in &self.allowed_verifier_digests {
            if verifier.0 == [0; 32]
                || previous_verifier.is_some_and(|old| old >= verifier.0)
            {
                return Err(QualificationWitnessError::InvalidWitnessPolicy);
            }
            previous_verifier = Some(verifier.0);
        }
        Ok(())
    }

    pub fn digest(&self) -> Result<Digest32, QualificationWitnessError> {
        self.validate()?;
        let mut transcript = Transcript::new(POLICY_DOMAIN);
        transcript.u16(self.schema_version);
        transcript.fixed(&self.policy_id);
        transcript.u64(self.witness_epoch);
        transcript.u16(self.threshold);
        transcript.u16(self.minimum_organizations);
        transcript.u16(self.minimum_services);
        transcript.u32(u32::try_from(self.allowed_verifier_digests.len()).map_err(|_| QualificationWitnessError::Encoding)?);
        for verifier in &self.allowed_verifier_digests {
            transcript.fixed(&verifier.0);
        }
        transcript.u32(u32::try_from(self.witnesses.len()).map_err(|_| QualificationWitnessError::Encoding)?);
        for witness in &self.witnesses {
            transcript.fixed(&witness.witness_id);
            transcript.fixed(&witness.organization_id);
            transcript.fixed(&witness.service_id);
            transcript.fixed(&witness.public_key);
        }
        Ok(Digest32(transcript.finish()))
    }

    fn witness(&self, witness_id: &[u8; 16]) -> Option<&QualificationWitnessIdentityV1> {
        self.witnesses
            .binary_search_by_key(witness_id, |witness| witness.witness_id)
            .ok()
            .map(|index| &self.witnesses[index])
    }

    fn allows_verifier(&self, digest: Digest32) -> bool {
        self.allowed_verifier_digests
            .binary_search_by_key(&digest.0, |item| item.0)
            .is_ok()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct QualificationWitnessAttestationV1 {
    pub schema_version: u16,
    pub signature_algorithm: String,
    pub policy_digest: Digest32,
    pub acceptance_digest: Digest32,
    pub verifier_digest: Digest32,
    pub witness_epoch: u64,
    pub witness_id: [u8; 16],
    pub organization_id: [u8; 16],
    pub service_id: [u8; 16],
    /// Monotonic sequence supplied by the witness service's own durable domain.
    /// V1 binds it into the signature but does not itself persist anti-rollback
    /// state; a higher-level witness service must reject sequence regression.
    pub witness_sequence: u64,
    pub signature: Vec<u8>,
}

impl QualificationWitnessAttestationV1 {
    fn unsigned_message(&self) -> Result<Vec<u8>, QualificationWitnessError> {
        if self.schema_version != WITNESS_SCHEMA_VERSION
            || self.signature_algorithm != SIGNATURE_ALGORITHM
            || self.policy_digest.0 == [0; 32]
            || self.acceptance_digest.0 == [0; 32]
            || self.verifier_digest.0 == [0; 32]
            || self.witness_epoch == 0
            || self.witness_id == [0; 16]
            || self.organization_id == [0; 16]
            || self.service_id == [0; 16]
            || self.witness_sequence == 0
        {
            return Err(QualificationWitnessError::InvalidAttestation);
        }
        let mut transcript = Transcript::new(ATTESTATION_DOMAIN);
        transcript.u16(self.schema_version);
        transcript.bytes(self.signature_algorithm.as_bytes())?;
        transcript.fixed(&self.policy_digest.0);
        transcript.fixed(&self.acceptance_digest.0);
        transcript.fixed(&self.verifier_digest.0);
        transcript.u64(self.witness_epoch);
        transcript.fixed(&self.witness_id);
        transcript.fixed(&self.organization_id);
        transcript.fixed(&self.service_id);
        transcript.u64(self.witness_sequence);
        Ok(transcript.into_bytes())
    }
}

/// Sign one already-verified release acceptance with an enrolled witness key.
///
/// Key storage is intentionally out of scope. Production callers should obtain
/// `SigningKey` from a dedicated witness/key-management boundary rather than
/// storing witness seeds in this crate or qualification evidence.
pub fn sign_qualification_acceptance_v1(
    acceptance: &QualificationAcceptanceV1,
    verifier_digest: Digest32,
    policy: &QualificationWitnessPolicyV1,
    witness_id: [u8; 16],
    witness_sequence: u64,
    signing_key: &SigningKey,
) -> Result<QualificationWitnessAttestationV1, QualificationWitnessError> {
    policy.validate()?;
    if verifier_digest.0 == [0; 32] || !policy.allows_verifier(verifier_digest) {
        return Err(QualificationWitnessError::VerifierNotAllowed);
    }
    let witness = policy
        .witness(&witness_id)
        .ok_or(QualificationWitnessError::WitnessNotAllowed)?;
    if signing_key.verifying_key().to_bytes() != witness.public_key {
        return Err(QualificationWitnessError::WitnessKeyMismatch);
    }
    if witness_sequence == 0 {
        return Err(QualificationWitnessError::InvalidAttestation);
    }

    let mut attestation = QualificationWitnessAttestationV1 {
        schema_version: WITNESS_SCHEMA_VERSION,
        signature_algorithm: SIGNATURE_ALGORITHM.to_string(),
        policy_digest: policy.digest()?,
        acceptance_digest: acceptance.digest()?,
        verifier_digest,
        witness_epoch: policy.witness_epoch,
        witness_id: witness.witness_id,
        organization_id: witness.organization_id,
        service_id: witness.service_id,
        witness_sequence,
        signature: Vec::new(),
    };
    attestation.signature = signing_key.sign(&attestation.unsigned_message()?).to_bytes().to_vec();
    Ok(attestation)
}

/// Opaque proof that the reviewed witness policy accepted a quorum for one
/// exact release-bound qualification acceptance.
#[derive(Debug)]
pub struct VerifiedQualificationWitnessQuorumV1 {
    acceptance_digest: Digest32,
    verifier_digest: Digest32,
    policy_digest: Digest32,
    witness_count: u16,
    organization_count: u16,
    service_count: u16,
}

impl VerifiedQualificationWitnessQuorumV1 {
    pub fn acceptance_digest(&self) -> Digest32 {
        self.acceptance_digest
    }

    pub fn verifier_digest(&self) -> Digest32 {
        self.verifier_digest
    }

    pub fn policy_digest(&self) -> Digest32 {
        self.policy_digest
    }

    pub fn witness_count(&self) -> u16 {
        self.witness_count
    }
}

pub fn verify_qualification_witness_quorum_v1(
    acceptance: &QualificationAcceptanceV1,
    verifier_digest: Digest32,
    policy: &QualificationWitnessPolicyV1,
    attestations: &[QualificationWitnessAttestationV1],
) -> Result<VerifiedQualificationWitnessQuorumV1, QualificationWitnessError> {
    policy.validate()?;
    if verifier_digest.0 == [0; 32] || !policy.allows_verifier(verifier_digest) {
        return Err(QualificationWitnessError::VerifierNotAllowed);
    }
    if attestations.is_empty() || attestations.len() > policy.witnesses.len() {
        return Err(QualificationWitnessError::QuorumNotSatisfied);
    }

    let policy_digest = policy.digest()?;
    let acceptance_digest = acceptance.digest()?;
    let mut seen_witnesses = BTreeSet::new();
    let mut organizations = BTreeSet::new();
    let mut services = BTreeSet::new();

    for attestation in attestations {
        if attestation.schema_version != WITNESS_SCHEMA_VERSION
            || attestation.signature_algorithm != SIGNATURE_ALGORITHM
            || attestation.policy_digest != policy_digest
            || attestation.acceptance_digest != acceptance_digest
            || attestation.verifier_digest != verifier_digest
            || attestation.witness_epoch != policy.witness_epoch
            || attestation.witness_sequence == 0
            || attestation.signature.len() != 64
        {
            return Err(QualificationWitnessError::InvalidAttestation);
        }
        if !seen_witnesses.insert(attestation.witness_id) {
            return Err(QualificationWitnessError::DuplicateWitness);
        }
        let witness = policy
            .witness(&attestation.witness_id)
            .ok_or(QualificationWitnessError::WitnessNotAllowed)?;
        if attestation.organization_id != witness.organization_id
            || attestation.service_id != witness.service_id
        {
            return Err(QualificationWitnessError::WitnessIdentityMismatch);
        }
        let key = VerifyingKey::from_bytes(&witness.public_key)
            .map_err(|_| QualificationWitnessError::InvalidWitnessIdentity)?;
        let signature = Signature::from_slice(&attestation.signature)
            .map_err(|_| QualificationWitnessError::InvalidSignature)?;
        key.verify_strict(&attestation.unsigned_message()?, &signature)
            .map_err(|_| QualificationWitnessError::InvalidSignature)?;
        organizations.insert(witness.organization_id);
        services.insert(witness.service_id);
    }

    if seen_witnesses.len() < usize::from(policy.threshold)
        || organizations.len() < usize::from(policy.minimum_organizations)
        || services.len() < usize::from(policy.minimum_services)
    {
        return Err(QualificationWitnessError::QuorumNotSatisfied);
    }

    Ok(VerifiedQualificationWitnessQuorumV1 {
        acceptance_digest,
        verifier_digest,
        policy_digest,
        witness_count: u16::try_from(seen_witnesses.len()).map_err(|_| QualificationWitnessError::Encoding)?,
        organization_count: u16::try_from(organizations.len()).map_err(|_| QualificationWitnessError::Encoding)?,
        service_count: u16::try_from(services.len()).map_err(|_| QualificationWitnessError::Encoding)?,
    })
}

fn parse_digest32(value: &str) -> Result<Digest32, QualificationWitnessError> {
    let mut out = [0u8; 32];
    if value.len() != 64
        || value.bytes().any(|byte| !byte.is_ascii_hexdigit() || byte.is_ascii_uppercase())
        || hex::decode_to_slice(value, &mut out).is_err()
        || out == [0; 32]
    {
        return Err(QualificationWitnessError::InvalidAcceptance);
    }
    Ok(Digest32(out))
}

fn parse_hex20(value: &str) -> Result<[u8; 20], QualificationWitnessError> {
    let mut out = [0u8; 20];
    if value.len() != 40
        || value.bytes().any(|byte| !byte.is_ascii_hexdigit() || byte.is_ascii_uppercase())
        || hex::decode_to_slice(value, &mut out).is_err()
        || out == [0; 20]
    {
        return Err(QualificationWitnessError::InvalidAcceptance);
    }
    Ok(out)
}

fn canonical_json_bytes(value: &Value) -> Result<Vec<u8>, QualificationWitnessError> {
    let mut out = Vec::new();
    canonical_json_into(value, &mut out)?;
    Ok(out)
}

fn canonical_json_into(value: &Value, out: &mut Vec<u8>) -> Result<(), QualificationWitnessError> {
    if out.len() > MAX_NIXPKGS_LOCKED_CANONICAL_BYTES {
        return Err(QualificationWitnessError::InvalidAcceptance);
    }
    match value {
        Value::Null => out.extend_from_slice(b"null"),
        Value::Bool(value) => out.extend_from_slice(if *value { b"true" } else { b"false" }),
        Value::Number(number) => {
            if !(number.is_i64() || number.is_u64()) {
                return Err(QualificationWitnessError::InvalidAcceptance);
            }
            out.extend_from_slice(number.to_string().as_bytes());
        }
        Value::String(value) => {
            let encoded = serde_json::to_string(value).map_err(|_| QualificationWitnessError::Encoding)?;
            out.extend_from_slice(encoded.as_bytes());
        }
        Value::Array(values) => {
            out.push(b'[');
            for (index, item) in values.iter().enumerate() {
                if index != 0 {
                    out.push(b',');
                }
                canonical_json_into(item, out)?;
            }
            out.push(b']');
        }
        Value::Object(values) => {
            out.push(b'{');
            let mut keys = values.keys().collect::<Vec<_>>();
            keys.sort_unstable();
            for (index, key) in keys.iter().enumerate() {
                if index != 0 {
                    out.push(b',');
                }
                let encoded_key = serde_json::to_string(key).map_err(|_| QualificationWitnessError::Encoding)?;
                out.extend_from_slice(encoded_key.as_bytes());
                out.push(b':');
                canonical_json_into(&values[*key], out)?;
            }
            out.push(b'}');
        }
    }
    if out.len() > MAX_NIXPKGS_LOCKED_CANONICAL_BYTES {
        return Err(QualificationWitnessError::InvalidAcceptance);
    }
    Ok(())
}

#[derive(Debug, Error)]
pub enum QualificationWitnessError {
    #[error("qualification acceptance is malformed")]
    InvalidAcceptance,
    #[error("qualification acceptance is not independently release-bound")]
    AcceptanceNotReleaseBound,
    #[error("witness identity is malformed")]
    InvalidWitnessIdentity,
    #[error("witness policy is malformed")]
    InvalidWitnessPolicy,
    #[error("evidence verifier implementation is not allowed by witness policy")]
    VerifierNotAllowed,
    #[error("witness is not enrolled by policy")]
    WitnessNotAllowed,
    #[error("signing key does not match enrolled witness identity")]
    WitnessKeyMismatch,
    #[error("witness identity metadata does not match policy")]
    WitnessIdentityMismatch,
    #[error("witness attestation is malformed or mismatched")]
    InvalidAttestation,
    #[error("witness signature is invalid")]
    InvalidSignature,
    #[error("duplicate witness attestation")]
    DuplicateWitness,
    #[error("witness quorum/diversity policy is not satisfied")]
    QuorumNotSatisfied,
    #[error("canonical encoding failed")]
    Encoding,
}

struct Transcript {
    bytes: Vec<u8>,
}

impl Transcript {
    fn new(domain: &[u8]) -> Self {
        let mut bytes = Vec::with_capacity(domain.len() + 512);
        bytes.extend_from_slice(domain);
        Self { bytes }
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

    fn bytes(&mut self, value: &[u8]) -> Result<(), QualificationWitnessError> {
        self.u32(u32::try_from(value.len()).map_err(|_| QualificationWitnessError::Encoding)?);
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
    use serde_json::json;

    fn acceptance_json(release_bound: bool) -> Vec<u8> {
        serde_json::to_vec(&json!({
            "schema": ACCEPTANCE_SCHEMA,
            "accepted": true,
            "qualification_result": "PASS",
            "archive_sha256": "11".repeat(32),
            "archive_hash_source": if release_bound { "caller" } else { "sidecar-unanchored" },
            "manifest_sha256": "12".repeat(32),
            "head": "13".repeat(20),
            "tree": "14".repeat(20),
            "external_head_bound": release_bound,
            "external_tree_bound": release_bound,
            "release_bound": release_bound,
            "nixpkgs_locked": {
                "type": "github",
                "owner": "NixOS",
                "repo": "nixpkgs",
                "rev": "abc123",
                "narHash": "sha256-example"
            },
            "flake_lock_sha256": "15".repeat(32),
            "rust_toolchain_sha256": "16".repeat(32),
            "approved_pcr_profile": "17".repeat(32),
            "policy_digest": "18".repeat(32),
            "ak_public_digest": "19".repeat(32),
            "challenge_digest": "1a".repeat(32),
            "probe_sha256": "1b".repeat(32),
            "quote_wrapper_sha256": "1c".repeat(32),
            "checkquote_wrapper_sha256": "1d".repeat(32),
            "verifier_store": "/nix/store/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa-symthaea-tpm2-verifier-v1"
        })).unwrap()
    }

    fn key(seed: u8) -> SigningKey {
        SigningKey::from_bytes(&[seed; 32])
    }

    fn identity(id: u8, org: u8, service: u8, key: &SigningKey) -> QualificationWitnessIdentityV1 {
        QualificationWitnessIdentityV1 {
            witness_id: [id; 16],
            organization_id: [org; 16],
            service_id: [service; 16],
            public_key: key.verifying_key().to_bytes(),
        }
    }

    fn policy(keys: &[SigningKey]) -> QualificationWitnessPolicyV1 {
        QualificationWitnessPolicyV1 {
            schema_version: WITNESS_SCHEMA_VERSION,
            policy_id: [0x41; 16],
            witness_epoch: 7,
            threshold: 2,
            minimum_organizations: 2,
            minimum_services: 2,
            allowed_verifier_digests: vec![Digest32([0x44; 32])],
            witnesses: vec![
                identity(1, 1, 1, &keys[0]),
                identity(2, 2, 2, &keys[1]),
                identity(3, 3, 3, &keys[2]),
            ],
        }
    }

    #[test]
    fn unanchored_acceptance_cannot_be_upgraded_by_witnesses() {
        assert!(matches!(
            parse_release_acceptance_v1(&acceptance_json(false)),
            Err(QualificationWitnessError::AcceptanceNotReleaseBound)
        ));
    }

    #[test]
    fn nixpkgs_object_key_order_does_not_change_acceptance_digest() {
        let first = parse_release_acceptance_v1(&acceptance_json(true)).unwrap();
        let mut raw: Value = serde_json::from_slice(&acceptance_json(true)).unwrap();
        raw["nixpkgs_locked"] = serde_json::from_str(
            r#"{"rev":"abc123","repo":"nixpkgs","owner":"NixOS","type":"github","narHash":"sha256-example"}"#,
        )
        .unwrap();
        let second = parse_release_acceptance_v1(&serde_json::to_vec(&raw).unwrap()).unwrap();
        assert_eq!(first.digest().unwrap(), second.digest().unwrap());
    }

    #[test]
    fn two_independent_witnesses_satisfy_threshold_and_diversity() {
        let acceptance = parse_release_acceptance_v1(&acceptance_json(true)).unwrap();
        let keys = [key(1), key(2), key(3)];
        let policy = policy(&keys);
        let verifier = Digest32([0x44; 32]);
        let a = sign_qualification_acceptance_v1(&acceptance, verifier, &policy, [1; 16], 10, &keys[0]).unwrap();
        let b = sign_qualification_acceptance_v1(&acceptance, verifier, &policy, [2; 16], 20, &keys[1]).unwrap();
        let verified = verify_qualification_witness_quorum_v1(&acceptance, verifier, &policy, &[a, b]).unwrap();
        assert_eq!(verified.witness_count(), 2);
        assert_eq!(verified.acceptance_digest(), acceptance.digest().unwrap());
    }

    #[test]
    fn one_witness_cannot_satisfy_two_of_two_policy() {
        let acceptance = parse_release_acceptance_v1(&acceptance_json(true)).unwrap();
        let keys = [key(1), key(2), key(3)];
        let policy = policy(&keys);
        let verifier = Digest32([0x44; 32]);
        let a = sign_qualification_acceptance_v1(&acceptance, verifier, &policy, [1; 16], 10, &keys[0]).unwrap();
        assert!(matches!(
            verify_qualification_witness_quorum_v1(&acceptance, verifier, &policy, &[a]),
            Err(QualificationWitnessError::QuorumNotSatisfied)
        ));
    }

    #[test]
    fn two_witnesses_from_one_organization_fail_diversity() {
        let acceptance = parse_release_acceptance_v1(&acceptance_json(true)).unwrap();
        let keys = [key(1), key(2), key(3)];
        let mut policy = policy(&keys);
        policy.witnesses[1].organization_id = [1; 16];
        // The policy itself has enough organizations because witness 3 remains
        // distinct, but attestations 1 + 2 do not satisfy runtime diversity.
        policy.validate().unwrap();
        let verifier = Digest32([0x44; 32]);
        let a = sign_qualification_acceptance_v1(&acceptance, verifier, &policy, [1; 16], 10, &keys[0]).unwrap();
        let b = sign_qualification_acceptance_v1(&acceptance, verifier, &policy, [2; 16], 20, &keys[1]).unwrap();
        assert!(matches!(
            verify_qualification_witness_quorum_v1(&acceptance, verifier, &policy, &[a, b]),
            Err(QualificationWitnessError::QuorumNotSatisfied)
        ));
    }

    #[test]
    fn changed_acceptance_invalidates_old_signatures() {
        let acceptance = parse_release_acceptance_v1(&acceptance_json(true)).unwrap();
        let keys = [key(1), key(2), key(3)];
        let policy = policy(&keys);
        let verifier = Digest32([0x44; 32]);
        let a = sign_qualification_acceptance_v1(&acceptance, verifier, &policy, [1; 16], 10, &keys[0]).unwrap();
        let b = sign_qualification_acceptance_v1(&acceptance, verifier, &policy, [2; 16], 20, &keys[1]).unwrap();

        let mut changed: Value = serde_json::from_slice(&acceptance_json(true)).unwrap();
        changed["archive_sha256"] = Value::String("31".repeat(32));
        let changed = parse_release_acceptance_v1(&serde_json::to_vec(&changed).unwrap()).unwrap();
        assert!(matches!(
            verify_qualification_witness_quorum_v1(&changed, verifier, &policy, &[a, b]),
            Err(QualificationWitnessError::InvalidAttestation)
        ));
    }

    #[test]
    fn verifier_implementation_is_part_of_witness_policy() {
        let acceptance = parse_release_acceptance_v1(&acceptance_json(true)).unwrap();
        let keys = [key(1), key(2), key(3)];
        let policy = policy(&keys);
        assert!(matches!(
            sign_qualification_acceptance_v1(
                &acceptance,
                Digest32([0x99; 32]),
                &policy,
                [1; 16],
                1,
                &keys[0],
            ),
            Err(QualificationWitnessError::VerifierNotAllowed)
        ));
    }

    #[test]
    fn policy_epoch_rotation_invalidates_prior_attestations() {
        let acceptance = parse_release_acceptance_v1(&acceptance_json(true)).unwrap();
        let keys = [key(1), key(2), key(3)];
        let policy = policy(&keys);
        let verifier = Digest32([0x44; 32]);
        let a = sign_qualification_acceptance_v1(&acceptance, verifier, &policy, [1; 16], 10, &keys[0]).unwrap();
        let b = sign_qualification_acceptance_v1(&acceptance, verifier, &policy, [2; 16], 20, &keys[1]).unwrap();

        let mut rotated = policy.clone();
        rotated.witness_epoch += 1;
        assert!(matches!(
            verify_qualification_witness_quorum_v1(&acceptance, verifier, &rotated, &[a, b]),
            Err(QualificationWitnessError::InvalidAttestation)
        ));
    }
}
