// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Series 20: a secret-free Ed25519 public-verification bundle.
//!
//! This is the prerequisite `checkpoint_hybrid_public_verifiability.rs`'s own doc comment
//! refers to ("Series 20 established a secret-free Ed25519 public-verification bundle. This
//! module adds an ML-DSA-65 overlay...") but that was never actually delivered in the
//! `12ff3e5c88` patch-series commit -- confirmed via `git log --all -S` before writing this
//! (see `TRACK_B_RECOVERY_PLAN_2026-07-30.md`). The API surface below was reverse-derived
//! from real usage across the 10 dependent checkpoint modules (`checkpoint_gossip_archive.rs`,
//! `checkpoint_gossip_transport.rs`, `checkpoint_hardware_signing.rs`,
//! `checkpoint_hybrid_public_verifiability.rs`, `checkpoint_trusted_time.rs`, and the
//! transparency-log family), not guessed from the "Series 20" name alone -- every method
//! signature and field name below is load-bearing for at least one real call site.
//!
//! "Secret-free" describes the VERIFICATION side: `CheckpointPublicVerifyingKey` and
//! `CheckpointPublicSignature` carry no secret material, so they can be freely gossiped,
//! archived, and checked by any party. The signing side (`CheckpointPublicSigningKey`) does
//! hold a secret (an Ed25519 seed) and is `ZeroizeOnDrop` (ed25519-dalek's `SigningKey`
//! zeroizes by default -- confirmed against its own `[features] default = [...,"zeroize"]`
//! before relying on this, not assumed).
//!
//! Domain separation follows the exact transcript convention already established (and
//! presumably already reviewed) in `checkpoint_hybrid_public_verifiability.rs`'s
//! `hybrid_domain_separated_message`: a little-endian u64 domain length, the domain bytes,
//! then the message bytes -- signed as one transcript so a signature over one domain can
//! never be replayed as valid for a different domain.

use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};

/// Cap on any single serialized artifact this module will sign/verify over. Matches the
/// bound every dependent file's own local digest helper already enforces before calling into
/// this module (each has its own `encoded.len() > MAX_CHECKPOINT_PUBLIC_ARTIFACT_BYTES`
/// check) -- 1 MiB is generous for the checkpoint-metadata-sized artifacts this subsystem
/// actually handles (receipts, policies, digests), not raw model weights.
pub const MAX_CHECKPOINT_PUBLIC_ARTIFACT_BYTES: usize = 1 << 20;

/// Cap on a signature domain separator string. Domains here are short, human-readable byte
/// constants (e.g. `GOSSIP_ARCHIVE_RECEIPT_SIGNATURE_DOMAIN`), never user-controlled data.
pub const MAX_CHECKPOINT_PUBLIC_SIGNATURE_DOMAIN_BYTES: usize = 256;

/// Cap on signers in a single [`CheckpointPublicVerificationBundle`]. Matches the
/// `MAX_CHECKPOINT_HYBRID_SIGNERS = 128` precedent already established in
/// `checkpoint_hybrid_public_verifiability.rs`.
pub const MAX_CHECKPOINT_PUBLIC_SIGNERS: usize = 128;

const ED25519_PUBLIC_KEY_BYTES: usize = 32;
const ED25519_SIGNATURE_BYTES: usize = 64;

pub const CHECKPOINT_PUBLIC_VERIFICATION_BUNDLE_SCHEMA: &str =
    "symthaea.checkpoint-series20-public-verification-bundle.v1";
pub const CHECKPOINT_PUBLIC_VERIFICATION_SUMMARY_SCHEMA: &str =
    "symthaea.checkpoint-series20-public-verification-summary.v1";

/// A 16-byte identifier for one Ed25519 keypair. Reject-all-zero is enforced at construction
/// (matches every sibling `Checkpoint*KeyId` type in this crate, e.g.
/// `CheckpointMlDsa65KeyId`), and `.0` is a public field -- every dependent file compares it
/// directly (`self.key_id.0 == [0u8; 16]`), not through an accessor.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CheckpointPublicKeyId(pub [u8; 16]);

impl CheckpointPublicKeyId {
    pub fn new(bytes: [u8; 16]) -> Result<Self, CheckpointPublicVerificationError> {
        if bytes == [0u8; 16] {
            return Err(CheckpointPublicVerificationError::InvalidKeyId);
        }
        Ok(Self(bytes))
    }
}

/// A secret Ed25519 signing key, paired with its own [`CheckpointPublicKeyId`]. Holds secret
/// material -- `ed25519_dalek::SigningKey` zeroizes its seed on drop by default.
pub struct CheckpointPublicSigningKey {
    key_id: CheckpointPublicKeyId,
    inner: SigningKey,
}

impl CheckpointPublicSigningKey {
    /// Derive a signing key from a 32-byte seed. The seed is the caller's responsibility to
    /// generate with a real CSPRNG (e.g. `ed25519_dalek::SigningKey::generate` upstream, or
    /// this crate's own genesis-seeded RNG for deterministic test fixtures) -- this
    /// constructor only rejects the obviously-degenerate all-zero seed, it cannot verify
    /// entropy quality.
    pub fn from_seed(
        key_id: CheckpointPublicKeyId,
        seed: [u8; 32],
    ) -> Result<Self, CheckpointPublicVerificationError> {
        if seed == [0u8; 32] {
            return Err(CheckpointPublicVerificationError::InvalidSeed);
        }
        Ok(Self {
            key_id,
            inner: SigningKey::from_bytes(&seed),
        })
    }

    pub fn key_id(&self) -> CheckpointPublicKeyId {
        self.key_id
    }

    pub fn verifying_key(&self) -> CheckpointPublicVerifyingKey {
        CheckpointPublicVerifyingKey {
            key_id: self.key_id,
            verifying_key_bytes: self.inner.verifying_key().to_bytes().to_vec(),
        }
    }

    /// Sign `message` under `domain`. Both are length-bounded and domain-separated into one
    /// transcript before signing (see module doc) -- a signature over one domain is never
    /// valid evidence for another.
    pub fn sign(
        &self,
        domain: &[u8],
        message: &[u8],
    ) -> Result<CheckpointPublicSignature, CheckpointPublicVerificationError> {
        validate_domain_and_message(domain, message)?;
        let transcript = domain_separated_transcript(domain, message);
        let signature = self.inner.sign(&transcript);
        Ok(CheckpointPublicSignature {
            key_id: self.key_id,
            signature_bytes: signature.to_bytes().to_vec(),
        })
    }
}

/// A secret-free Ed25519 verifying key. Every field is public and every dependent file
/// accesses them directly (`archive.verifying_key.key_id`,
/// `observer.verifying_key.verifying_key_bytes`) rather than through accessors.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointPublicVerifyingKey {
    pub key_id: CheckpointPublicKeyId,
    pub verifying_key_bytes: Vec<u8>,
}

impl CheckpointPublicVerifyingKey {
    /// Confirms this is a well-formed, non-degenerate Ed25519 verifying key: right length,
    /// not all-zero, and actually decodes as a valid compressed Edwards point (the
    /// `VerifyingKey::from_bytes` call below rejects points not on the curve / not in the
    /// prime-order subgroup, per ed25519-dalek's own validation).
    pub fn validate(&self) -> Result<(), CheckpointPublicVerificationError> {
        if self.key_id.0 == [0u8; 16]
            || self.verifying_key_bytes.len() != ED25519_PUBLIC_KEY_BYTES
            || self.verifying_key_bytes.iter().all(|byte| *byte == 0)
        {
            return Err(CheckpointPublicVerificationError::InvalidVerifyingKey);
        }
        let bytes = as_ed25519_public_key_bytes(&self.verifying_key_bytes)?;
        VerifyingKey::from_bytes(&bytes)
            .map_err(|_| CheckpointPublicVerificationError::InvalidVerifyingKey)?;
        Ok(())
    }

    /// Verify `signature` was produced by the matching signing key over `domain`/`message`.
    pub fn verify(
        &self,
        domain: &[u8],
        message: &[u8],
        signature: &CheckpointPublicSignature,
    ) -> Result<(), CheckpointPublicVerificationError> {
        self.validate()?;
        validate_domain_and_message(domain, message)?;
        if signature.key_id != self.key_id {
            return Err(CheckpointPublicVerificationError::WrongKey);
        }
        let public_key_bytes = as_ed25519_public_key_bytes(&self.verifying_key_bytes)?;
        let verifying_key = VerifyingKey::from_bytes(&public_key_bytes)
            .map_err(|_| CheckpointPublicVerificationError::InvalidVerifyingKey)?;
        let signature_bytes = as_ed25519_signature_bytes(&signature.signature_bytes)?;
        let signature = Signature::from_bytes(&signature_bytes);
        let transcript = domain_separated_transcript(domain, message);
        verifying_key
            .verify(&transcript, &signature)
            .map_err(|_| CheckpointPublicVerificationError::VerificationFailed)
    }
}

/// A secret-free Ed25519 signature, tagged with the [`CheckpointPublicKeyId`] of the signer
/// that produced it (checked against the verifying key's own id in
/// [`CheckpointPublicVerifyingKey::verify`], so a signature can never be silently checked
/// against the wrong key).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointPublicSignature {
    pub key_id: CheckpointPublicKeyId,
    pub signature_bytes: Vec<u8>,
}

/// A self-contained, time-bounded, threshold multi-signer verification bundle over one
/// artifact digest. Used by `checkpoint_hybrid_public_verifiability.rs`'s
/// `CheckpointHybridVerificationBundle.classical_bundle` field as the classical (non-PQ) half
/// of its hybrid verification, matching the schema/`.verify(unix_seconds)`/summary pattern
/// already established by every sibling bundle type in this crate family (e.g.
/// `CheckpointGossipArchiveBundle`, `CheckpointHybridVerificationBundle` itself).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointPublicVerificationBundle {
    pub schema: String,
    pub artifact_digest: [u8; 32],
    pub signature_domain: Vec<u8>,
    pub signers: Vec<CheckpointPublicVerifyingKey>,
    pub signatures: Vec<CheckpointPublicSignature>,
    pub threshold: usize,
    pub valid_from_unix_seconds: u64,
    pub valid_until_unix_seconds: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CheckpointPublicVerificationSummary {
    pub schema: String,
    pub artifact_digest: [u8; 32],
    pub valid_signatures: usize,
    pub unique_signers: usize,
}

impl CheckpointPublicVerificationBundle {
    /// Verify this bundle at `verification_time_unix_seconds`: schema, time bounds, signer
    /// count, distinct signature-per-signer, and that at least `threshold` signatures verify
    /// against the artifact digest under this bundle's own signature domain.
    pub fn verify(
        &self,
        verification_time_unix_seconds: u64,
    ) -> Result<CheckpointPublicVerificationSummary, CheckpointPublicVerificationError> {
        if self.schema != CHECKPOINT_PUBLIC_VERIFICATION_BUNDLE_SCHEMA
            || self.artifact_digest == [0u8; 32]
            || self.signers.is_empty()
            || self.signers.len() > MAX_CHECKPOINT_PUBLIC_SIGNERS
            || self.signatures.is_empty()
            || self.signatures.len() > MAX_CHECKPOINT_PUBLIC_SIGNERS
            || self.threshold == 0
            || self.threshold > self.signers.len()
            || self.valid_until_unix_seconds <= self.valid_from_unix_seconds
            || verification_time_unix_seconds < self.valid_from_unix_seconds
            || verification_time_unix_seconds > self.valid_until_unix_seconds
        {
            return Err(CheckpointPublicVerificationError::InvalidBundle);
        }

        let mut seen_signers = std::collections::HashSet::new();
        let mut valid_signatures = 0usize;
        for signature in &self.signatures {
            if !seen_signers.insert(signature.key_id) {
                // A signer appearing twice in the signature list can never legitimately
                // raise the threshold count -- reject outright rather than silently
                // de-duplicating, since a bundle constructor should never produce this.
                return Err(CheckpointPublicVerificationError::DuplicateSigner);
            }
            let Some(signer) = self
                .signers
                .iter()
                .find(|candidate| candidate.key_id == signature.key_id)
            else {
                return Err(CheckpointPublicVerificationError::WrongKey);
            };
            if signer
                .verify(&self.signature_domain, &self.artifact_digest, signature)
                .is_ok()
            {
                valid_signatures += 1;
            }
        }

        if valid_signatures < self.threshold {
            return Err(CheckpointPublicVerificationError::ThresholdNotMet);
        }

        Ok(CheckpointPublicVerificationSummary {
            schema: CHECKPOINT_PUBLIC_VERIFICATION_SUMMARY_SCHEMA.to_owned(),
            artifact_digest: self.artifact_digest,
            valid_signatures,
            unique_signers: seen_signers.len(),
        })
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CheckpointPublicVerificationError {
    InvalidKeyId,
    InvalidSeed,
    InvalidVerifyingKey,
    InvalidSignature,
    InvalidMessage,
    InvalidBundle,
    WrongKey,
    DuplicateSigner,
    ThresholdNotMet,
    VerificationFailed,
}

impl std::fmt::Display for CheckpointPublicVerificationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let msg = match self {
            Self::InvalidKeyId => "invalid (all-zero) checkpoint public key id",
            Self::InvalidSeed => "invalid (all-zero) checkpoint public signing key seed",
            Self::InvalidVerifyingKey => "invalid checkpoint public verifying key",
            Self::InvalidSignature => "invalid checkpoint public signature encoding",
            Self::InvalidMessage => "invalid checkpoint public signing domain or message",
            Self::InvalidBundle => "invalid checkpoint public verification bundle",
            Self::WrongKey => "checkpoint public signature key id does not match verifying key",
            Self::DuplicateSigner => "duplicate signer in checkpoint public verification bundle",
            Self::ThresholdNotMet => {
                "checkpoint public verification bundle did not meet its signature threshold"
            }
            Self::VerificationFailed => "checkpoint public signature verification failed",
        };
        f.write_str(msg)
    }
}

impl std::error::Error for CheckpointPublicVerificationError {}

fn validate_domain_and_message(
    domain: &[u8],
    message: &[u8],
) -> Result<(), CheckpointPublicVerificationError> {
    if domain.is_empty()
        || domain.len() > MAX_CHECKPOINT_PUBLIC_SIGNATURE_DOMAIN_BYTES
        || message.is_empty()
        || message.len() > MAX_CHECKPOINT_PUBLIC_ARTIFACT_BYTES
    {
        return Err(CheckpointPublicVerificationError::InvalidMessage);
    }
    Ok(())
}

/// Length-prefix the domain, then append it and the message -- so `domain || message` can
/// never be reinterpreted as a different `(domain, message)` split (a classic domain
/// separation pitfall). Mirrors `checkpoint_hybrid_public_verifiability.rs`'s
/// `hybrid_domain_separated_message` exactly.
fn domain_separated_transcript(domain: &[u8], message: &[u8]) -> Vec<u8> {
    let mut transcript = Vec::with_capacity(8 + domain.len() + message.len());
    transcript.extend_from_slice(&(domain.len() as u64).to_le_bytes());
    transcript.extend_from_slice(domain);
    transcript.extend_from_slice(message);
    transcript
}

fn as_ed25519_public_key_bytes(
    bytes: &[u8],
) -> Result<[u8; ED25519_PUBLIC_KEY_BYTES], CheckpointPublicVerificationError> {
    bytes
        .try_into()
        .map_err(|_| CheckpointPublicVerificationError::InvalidVerifyingKey)
}

fn as_ed25519_signature_bytes(
    bytes: &[u8],
) -> Result<[u8; ED25519_SIGNATURE_BYTES], CheckpointPublicVerificationError> {
    bytes
        .try_into()
        .map_err(|_| CheckpointPublicVerificationError::InvalidSignature)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn signing_key(id: u8, seed: u8) -> CheckpointPublicSigningKey {
        CheckpointPublicSigningKey::from_seed(
            CheckpointPublicKeyId::new([id; 16]).unwrap(),
            [seed; 32],
        )
        .unwrap()
    }

    fn bundle(
        signers: &[&CheckpointPublicSigningKey],
        domain: &[u8],
        artifact_digest: [u8; 32],
        threshold: usize,
    ) -> CheckpointPublicVerificationBundle {
        let verifying_keys: Vec<_> = signers.iter().map(|k| k.verifying_key()).collect();
        let signatures: Vec<_> = signers
            .iter()
            .map(|k| k.sign(domain, &artifact_digest).unwrap())
            .collect();
        CheckpointPublicVerificationBundle {
            schema: CHECKPOINT_PUBLIC_VERIFICATION_BUNDLE_SCHEMA.to_owned(),
            artifact_digest,
            signature_domain: domain.to_vec(),
            signers: verifying_keys,
            signatures,
            threshold,
            valid_from_unix_seconds: 100,
            valid_until_unix_seconds: 200,
        }
    }

    #[test]
    fn key_id_rejects_all_zero() {
        assert!(CheckpointPublicKeyId::new([0u8; 16]).is_err());
        assert!(CheckpointPublicKeyId::new([1u8; 16]).is_ok());
    }

    #[test]
    fn signing_key_rejects_all_zero_seed() {
        assert!(
            CheckpointPublicSigningKey::from_seed(
                CheckpointPublicKeyId::new([1u8; 16]).unwrap(),
                [0u8; 32]
            )
            .is_err()
        );
    }

    #[test]
    fn sign_and_verify_round_trip() {
        let key = signing_key(1, 7);
        let verifying_key = key.verifying_key();
        let signature = key.sign(b"test-domain", b"hello world").unwrap();
        assert!(
            verifying_key
                .verify(b"test-domain", b"hello world", &signature)
                .is_ok()
        );
    }

    #[test]
    fn verify_rejects_tampered_message() {
        let key = signing_key(1, 7);
        let verifying_key = key.verifying_key();
        let signature = key.sign(b"test-domain", b"hello world").unwrap();
        assert!(
            verifying_key
                .verify(b"test-domain", b"goodbye world", &signature)
                .is_err()
        );
    }

    #[test]
    fn verify_rejects_wrong_domain() {
        let key = signing_key(1, 7);
        let verifying_key = key.verifying_key();
        let signature = key.sign(b"domain-a", b"hello world").unwrap();
        assert!(
            verifying_key
                .verify(b"domain-b", b"hello world", &signature)
                .is_err(),
            "a signature over one domain must not verify under a different domain"
        );
    }

    #[test]
    fn verify_rejects_wrong_signer() {
        let key_a = signing_key(1, 7);
        let key_b = signing_key(2, 9);
        let signature = key_a.sign(b"test-domain", b"hello world").unwrap();
        assert!(
            key_b
                .verifying_key()
                .verify(b"test-domain", b"hello world", &signature)
                .is_err()
        );
    }

    #[test]
    fn verify_rejects_key_id_signature_mismatch() {
        let key = signing_key(1, 7);
        let mut signature = key.sign(b"test-domain", b"hello world").unwrap();
        // Forge the signature's claimed key id without re-signing -- must be rejected even
        // though the raw Ed25519 bytes are still cryptographically valid for the original key.
        signature.key_id = CheckpointPublicKeyId::new([9u8; 16]).unwrap();
        let verifying_key = key.verifying_key();
        assert_eq!(
            verifying_key.verify(b"test-domain", b"hello world", &signature),
            Err(CheckpointPublicVerificationError::WrongKey)
        );
    }

    #[test]
    fn verifying_key_validate_rejects_wrong_length() {
        let mut vk = signing_key(1, 7).verifying_key();
        vk.verifying_key_bytes.push(0);
        assert!(vk.validate().is_err());
    }

    #[test]
    fn verifying_key_validate_rejects_all_zero_bytes() {
        let vk = CheckpointPublicVerifyingKey {
            key_id: CheckpointPublicKeyId::new([1u8; 16]).unwrap(),
            verifying_key_bytes: vec![0u8; ED25519_PUBLIC_KEY_BYTES],
        };
        assert!(vk.validate().is_err());
    }

    #[test]
    fn sign_rejects_empty_domain_and_oversized_message() {
        let key = signing_key(1, 7);
        assert!(key.sign(b"", b"hello world").is_err());
        let oversized = vec![0u8; MAX_CHECKPOINT_PUBLIC_ARTIFACT_BYTES + 1];
        assert!(key.sign(b"domain", &oversized).is_err());
    }

    #[test]
    fn bundle_verifies_when_threshold_met() {
        let k1 = signing_key(1, 1);
        let k2 = signing_key(2, 2);
        let k3 = signing_key(3, 3);
        let digest = [42u8; 32];
        let b = bundle(&[&k1, &k2, &k3], b"bundle-domain", digest, 2);
        let summary = b.verify(150).unwrap();
        assert_eq!(summary.valid_signatures, 3);
        assert_eq!(summary.unique_signers, 3);
        assert_eq!(summary.artifact_digest, digest);
    }

    #[test]
    fn bundle_rejects_below_threshold() {
        let k1 = signing_key(1, 1);
        let k2 = signing_key(2, 2);
        let digest = [7u8; 32];
        // Threshold of 3 but only 2 signers ever provided -- construct directly, since
        // `bundle()`'s threshold param would be rejected at construction time by `verify()`'s
        // own `threshold > signers.len()` check; this tests a threshold that IS satisfiable
        // by signer count but not by valid signatures actually present.
        let mut b = bundle(&[&k1, &k2], b"bundle-domain", digest, 2);
        b.signatures.pop(); // drop one real signature -- only 1 of 2 required now present
        assert_eq!(
            b.verify(150),
            Err(CheckpointPublicVerificationError::ThresholdNotMet)
        );
    }

    #[test]
    fn bundle_rejects_outside_time_window() {
        let k1 = signing_key(1, 1);
        let b = bundle(&[&k1], b"bundle-domain", [1u8; 32], 1);
        assert!(b.verify(50).is_err(), "before valid_from must be rejected");
        assert!(b.verify(250).is_err(), "after valid_until must be rejected");
        assert!(b.verify(150).is_ok());
    }

    #[test]
    fn bundle_rejects_too_many_signers() {
        let signers: Vec<_> = (0..(MAX_CHECKPOINT_PUBLIC_SIGNERS as u8 + 1))
            .map(|i| signing_key(i.wrapping_add(1), i.wrapping_add(1)))
            .collect();
        let refs: Vec<&CheckpointPublicSigningKey> = signers.iter().collect();
        let b = bundle(&refs, b"bundle-domain", [3u8; 32], 1);
        assert_eq!(
            b.verify(150),
            Err(CheckpointPublicVerificationError::InvalidBundle)
        );
    }

    #[test]
    fn bundle_rejects_duplicate_signer_in_signatures() {
        let k1 = signing_key(1, 1);
        let digest = [5u8; 32];
        let mut b = bundle(&[&k1], b"bundle-domain", digest, 1);
        let dup = b.signatures[0].clone();
        b.signatures.push(dup);
        assert_eq!(
            b.verify(150),
            Err(CheckpointPublicVerificationError::DuplicateSigner)
        );
    }

    #[test]
    fn bundle_one_bad_signature_does_not_count_toward_threshold() {
        let k1 = signing_key(1, 1);
        let k2 = signing_key(2, 2);
        let digest = [8u8; 32];
        let mut b = bundle(&[&k1, &k2], b"bundle-domain", digest, 2);
        // Tamper one signature's bytes so it fails verification but still parses (right
        // length) -- confirms a corrupt-but-well-formed signature doesn't silently count.
        b.signatures[0].signature_bytes[0] ^= 0xFF;
        assert_eq!(
            b.verify(150),
            Err(CheckpointPublicVerificationError::ThresholdNotMet)
        );
    }
}
