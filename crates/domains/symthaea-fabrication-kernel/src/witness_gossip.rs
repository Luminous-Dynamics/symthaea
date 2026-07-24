// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Signed transparency-witness gossip and portable equivocation proofs.
//!
//! Witness quorum proves that several parties observed one checkpoint. Gossip
//! adds a cross-observer channel: a witness signs the checkpoint root it saw,
//! and any party can later prove that the same witness endorsed two different
//! roots for the same log size.

use crate::attestation::{DetachedSignature, SignatureAlgorithm};
use crate::crypto_digest::{Sha256, Sha256Digest};
use crate::trust::{KeyEligibility, KeyUsage, TrustSnapshot, digest_trust_snapshot};
use serde::{Deserialize, Serialize};

pub const WITNESS_GOSSIP_SCHEMA: &str = "symthaea.fabrication.witness-gossip.v1";
pub const SIGNED_WITNESS_GOSSIP_SCHEMA: &str = "symthaea.fabrication.signed-witness-gossip.v1";
pub const WITNESS_EQUIVOCATION_SCHEMA: &str = "symthaea.fabrication.witness-equivocation.v1";

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WitnessGossipStatement {
    pub schema_version: String,
    pub witness_organization: String,
    pub witness_region: String,
    pub checkpoint_log_size: u64,
    pub checkpoint_root_digest: Sha256Digest,
    pub checkpoint_digest: Sha256Digest,
    pub observed_at_unix_s: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SignedWitnessGossip {
    pub schema_version: String,
    pub statement: WitnessGossipStatement,
    pub statement_digest: Sha256Digest,
    pub signature: DetachedSignature,
}

pub trait WitnessGossipSigner {
    fn algorithm(&self) -> SignatureAlgorithm;
    fn key_id(&self) -> &str;
    fn sign_witness_gossip(&self, message: &[u8]) -> Result<Vec<u8>, String>;
}

pub trait WitnessGossipVerifier {
    fn verify_witness_gossip(
        &self,
        algorithm: &SignatureAlgorithm,
        key_id: &str,
        message: &[u8],
        signature: &[u8],
    ) -> Result<bool, String>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WitnessGossipPolicy {
    pub maximum_observation_age_s: u64,
    pub maximum_signature_bytes: usize,
}

impl Default for WitnessGossipPolicy {
    fn default() -> Self {
        Self {
            maximum_observation_age_s: 24 * 3_600,
            maximum_signature_bytes: 64 * 1024,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum WitnessGossipError {
    UnsupportedSchema,
    InvalidPolicy,
    InvalidIdentifier,
    InvalidLogSize,
    ObservationInFuture,
    ObservationStale,
    EmptySignature,
    SignatureTooLarge,
    InvalidAlgorithm,
    DigestMismatch,
    TrustSnapshotInvalid,
    TrustSnapshotStale,
    SignerUnknown,
    SignerNotYetValid,
    SignerExpired,
    SignerRetired,
    SignerRevoked,
    SignerUsageNotAllowed,
    InvalidSignature,
    VerificationProvider(String),
    SigningProvider(String),
    NotSameWitness,
    LogSizeMismatch,
    RootsDoNotConflict,
    InvalidProofTime,
    Encoding(String),
}

#[derive(Debug, Clone)]
pub struct VerifiedWitnessGossip {
    signed: SignedWitnessGossip,
    trust_snapshot_digest: Sha256Digest,
}

impl VerifiedWitnessGossip {
    pub fn signed(&self) -> &SignedWitnessGossip {
        &self.signed
    }
    pub fn statement(&self) -> &WitnessGossipStatement {
        &self.signed.statement
    }
    pub fn statement_digest(&self) -> Sha256Digest {
        self.signed.statement_digest
    }
    pub fn signer_algorithm(&self) -> &SignatureAlgorithm {
        &self.signed.signature.algorithm
    }
    pub fn signer_key_id(&self) -> &str {
        &self.signed.signature.key_id
    }
    pub fn trust_snapshot_digest(&self) -> Sha256Digest {
        self.trust_snapshot_digest
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct WitnessEquivocationProof {
    pub schema_version: String,
    pub signer_algorithm: SignatureAlgorithm,
    pub signer_key_id: String,
    pub witness_organization: String,
    pub checkpoint_log_size: u64,
    pub first_statement_digest: Sha256Digest,
    pub first_root_digest: Sha256Digest,
    pub second_statement_digest: Sha256Digest,
    pub second_root_digest: Sha256Digest,
    pub trust_snapshot_digest: Sha256Digest,
    pub proved_at_unix_s: u64,
}

#[derive(Debug, Clone)]
pub struct VerifiedWitnessEquivocation {
    proof: WitnessEquivocationProof,
    proof_digest: Sha256Digest,
}

impl VerifiedWitnessEquivocation {
    pub fn proof(&self) -> &WitnessEquivocationProof {
        &self.proof
    }
    pub fn proof_digest(&self) -> Sha256Digest {
        self.proof_digest
    }
}

pub fn sign_witness_gossip(
    statement: WitnessGossipStatement,
    signer: &dyn WitnessGossipSigner,
) -> Result<SignedWitnessGossip, WitnessGossipError> {
    validate_statement(&statement)?;
    if !signer.algorithm().is_canonical() {
        return Err(WitnessGossipError::InvalidAlgorithm);
    }
    validate_identifier(signer.key_id())?;
    let statement_digest = digest_witness_gossip_statement(&statement)?;
    let signature = signer
        .sign_witness_gossip(&gossip_signature_message(statement_digest))
        .map_err(WitnessGossipError::SigningProvider)?;
    if signature.is_empty() {
        return Err(WitnessGossipError::EmptySignature);
    }
    Ok(SignedWitnessGossip {
        schema_version: SIGNED_WITNESS_GOSSIP_SCHEMA.into(),
        statement,
        statement_digest,
        signature: DetachedSignature {
            algorithm: signer.algorithm(),
            key_id: signer.key_id().to_string(),
            signature,
        },
    })
}

pub fn digest_witness_gossip_statement(
    statement: &WitnessGossipStatement,
) -> Result<Sha256Digest, WitnessGossipError> {
    validate_statement(statement)?;
    digest_serialized(
        b"symthaea.fabrication.witness-gossip-statement-digest.v1\0",
        statement,
    )
}

pub fn verify_witness_gossip(
    signed: &SignedWitnessGossip,
    policy: &WitnessGossipPolicy,
    trust_snapshot: &TrustSnapshot,
    now_unix_s: u64,
    verifier: &dyn WitnessGossipVerifier,
) -> Result<VerifiedWitnessGossip, WitnessGossipError> {
    validate_policy(policy)?;
    if signed.schema_version != SIGNED_WITNESS_GOSSIP_SCHEMA {
        return Err(WitnessGossipError::UnsupportedSchema);
    }
    validate_statement(&signed.statement)?;
    if signed.statement.observed_at_unix_s > now_unix_s {
        return Err(WitnessGossipError::ObservationInFuture);
    }
    if now_unix_s.saturating_sub(signed.statement.observed_at_unix_s)
        > policy.maximum_observation_age_s
    {
        return Err(WitnessGossipError::ObservationStale);
    }
    if !signed.signature.algorithm.is_canonical() {
        return Err(WitnessGossipError::InvalidAlgorithm);
    }
    validate_identifier(&signed.signature.key_id)?;
    if signed.signature.signature.is_empty() {
        return Err(WitnessGossipError::EmptySignature);
    }
    if signed.signature.signature.len() > policy.maximum_signature_bytes {
        return Err(WitnessGossipError::SignatureTooLarge);
    }
    if digest_witness_gossip_statement(&signed.statement)? != signed.statement_digest {
        return Err(WitnessGossipError::DigestMismatch);
    }
    trust_snapshot
        .validate()
        .map_err(|_| WitnessGossipError::TrustSnapshotInvalid)?;
    if !trust_snapshot.is_fresh_at(now_unix_s) {
        return Err(WitnessGossipError::TrustSnapshotStale);
    }
    match trust_snapshot.key_eligibility(
        &signed.signature.algorithm,
        &signed.signature.key_id,
        KeyUsage::WitnessGossip,
        now_unix_s,
    ) {
        KeyEligibility::Eligible => {}
        KeyEligibility::Unknown => return Err(WitnessGossipError::SignerUnknown),
        KeyEligibility::NotYetValid => return Err(WitnessGossipError::SignerNotYetValid),
        KeyEligibility::Expired => return Err(WitnessGossipError::SignerExpired),
        KeyEligibility::Retired => return Err(WitnessGossipError::SignerRetired),
        KeyEligibility::Revoked => return Err(WitnessGossipError::SignerRevoked),
        KeyEligibility::UsageNotAllowed => return Err(WitnessGossipError::SignerUsageNotAllowed),
    }
    let valid = verifier
        .verify_witness_gossip(
            &signed.signature.algorithm,
            &signed.signature.key_id,
            &gossip_signature_message(signed.statement_digest),
            &signed.signature.signature,
        )
        .map_err(WitnessGossipError::VerificationProvider)?;
    if !valid {
        return Err(WitnessGossipError::InvalidSignature);
    }
    Ok(VerifiedWitnessGossip {
        signed: signed.clone(),
        trust_snapshot_digest: digest_trust_snapshot(trust_snapshot)
            .map_err(|_| WitnessGossipError::TrustSnapshotInvalid)?,
    })
}

pub fn prove_witness_equivocation(
    first: &VerifiedWitnessGossip,
    second: &VerifiedWitnessGossip,
    proved_at_unix_s: u64,
) -> Result<VerifiedWitnessEquivocation, WitnessGossipError> {
    if first.signer_algorithm() != second.signer_algorithm()
        || first.signer_key_id() != second.signer_key_id()
        || first.statement().witness_organization != second.statement().witness_organization
    {
        return Err(WitnessGossipError::NotSameWitness);
    }
    if first.statement().checkpoint_log_size != second.statement().checkpoint_log_size {
        return Err(WitnessGossipError::LogSizeMismatch);
    }
    if first.statement().checkpoint_root_digest == second.statement().checkpoint_root_digest {
        return Err(WitnessGossipError::RootsDoNotConflict);
    }
    if first.trust_snapshot_digest() != second.trust_snapshot_digest() {
        return Err(WitnessGossipError::TrustSnapshotInvalid);
    }
    if proved_at_unix_s < first.statement().observed_at_unix_s
        || proved_at_unix_s < second.statement().observed_at_unix_s
    {
        return Err(WitnessGossipError::InvalidProofTime);
    }
    let proof = WitnessEquivocationProof {
        schema_version: WITNESS_EQUIVOCATION_SCHEMA.into(),
        signer_algorithm: first.signer_algorithm().clone(),
        signer_key_id: first.signer_key_id().to_string(),
        witness_organization: first.statement().witness_organization.clone(),
        checkpoint_log_size: first.statement().checkpoint_log_size,
        first_statement_digest: first.statement_digest(),
        first_root_digest: first.statement().checkpoint_root_digest,
        second_statement_digest: second.statement_digest(),
        second_root_digest: second.statement().checkpoint_root_digest,
        trust_snapshot_digest: first.trust_snapshot_digest(),
        proved_at_unix_s,
    };
    let proof_digest = digest_witness_equivocation_proof(&proof)?;
    Ok(VerifiedWitnessEquivocation {
        proof,
        proof_digest,
    })
}

pub fn digest_witness_equivocation_proof(
    proof: &WitnessEquivocationProof,
) -> Result<Sha256Digest, WitnessGossipError> {
    if proof.schema_version != WITNESS_EQUIVOCATION_SCHEMA {
        return Err(WitnessGossipError::UnsupportedSchema);
    }
    if proof.first_root_digest == proof.second_root_digest {
        return Err(WitnessGossipError::RootsDoNotConflict);
    }
    if proof.checkpoint_log_size == 0 || proof.proved_at_unix_s == 0 {
        return Err(WitnessGossipError::InvalidProofTime);
    }
    validate_identifier(&proof.signer_key_id)?;
    validate_identifier(&proof.witness_organization)?;
    digest_serialized(
        b"symthaea.fabrication.witness-equivocation-proof-digest.v1\0",
        proof,
    )
}

fn validate_statement(statement: &WitnessGossipStatement) -> Result<(), WitnessGossipError> {
    if statement.schema_version != WITNESS_GOSSIP_SCHEMA {
        return Err(WitnessGossipError::UnsupportedSchema);
    }
    validate_identifier(&statement.witness_organization)?;
    validate_identifier(&statement.witness_region)?;
    if statement.checkpoint_log_size == 0 {
        return Err(WitnessGossipError::InvalidLogSize);
    }
    Ok(())
}

fn validate_policy(policy: &WitnessGossipPolicy) -> Result<(), WitnessGossipError> {
    if policy.maximum_observation_age_s == 0 || policy.maximum_signature_bytes == 0 {
        return Err(WitnessGossipError::InvalidPolicy);
    }
    Ok(())
}

fn validate_identifier(value: &str) -> Result<(), WitnessGossipError> {
    if value.trim().is_empty()
        || value != value.trim()
        || value.len() > 256
        || value.chars().any(char::is_control)
    {
        return Err(WitnessGossipError::InvalidIdentifier);
    }
    Ok(())
}

fn gossip_signature_message(digest: Sha256Digest) -> Vec<u8> {
    let mut message = b"symthaea.fabrication.witness-gossip-signature.v1\0".to_vec();
    message.extend_from_slice(&digest.0);
    message
}

fn digest_serialized<T: Serialize>(
    domain: &[u8],
    value: &T,
) -> Result<Sha256Digest, WitnessGossipError> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| WitnessGossipError::Encoding(error.to_string()))?;
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(&bytes);
    Ok(hasher.finalize())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn proof_digest_rejects_identical_roots() {
        let proof = WitnessEquivocationProof {
            schema_version: WITNESS_EQUIVOCATION_SCHEMA.into(),
            signer_algorithm: SignatureAlgorithm::Ed25519,
            signer_key_id: "witness-a".into(),
            witness_organization: "org-a".into(),
            checkpoint_log_size: 8,
            first_statement_digest: Sha256Digest([1; 32]),
            first_root_digest: Sha256Digest([2; 32]),
            second_statement_digest: Sha256Digest([3; 32]),
            second_root_digest: Sha256Digest([2; 32]),
            trust_snapshot_digest: Sha256Digest([4; 32]),
            proved_at_unix_s: 100,
        };
        assert_eq!(
            digest_witness_equivocation_proof(&proof),
            Err(WitnessGossipError::RootsDoNotConflict)
        );
    }
}
