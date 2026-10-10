// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Strict Ed25519 verification for calibration review receipts.
//!
//! This module verifies a canonical signed receipt against an explicitly
//! configured trust store, its key lifecycle, and a caller-supplied current time.
//! A trust store is an authority input: deployments must authenticate and govern
//! its distribution, sequence, freshness, and revocation updates independently.

use ed25519_dalek::{Signature, VerifyingKey};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use crate::{ArtifactReference, VerifiedArtifactBytes};

pub const CALIBRATION_REVIEW_SCHEMA: &str = "symthaea.calibration-review.v1";
pub const REVIEW_TRUST_STORE_SCHEMA: &str = "symthaea.calibration-review-trust.v1";
pub const MAX_REVIEW_KEY_ID_BYTES: usize = 256;
pub const MAX_REVIEW_RECORD_ID_BYTES: usize = 256;
pub const MAX_REVIEW_TRUST_KEYS: usize = 4096;

/// Review authority input with an explicit public key lifecycle.
#[derive(Debug, Clone)]
pub struct ReviewTrustKey {
    key_id: String,
    verifying_key: VerifyingKey,
    not_before_unix_s: u64,
    not_after_unix_s: Option<u64>,
    revoked_at_unix_s: Option<u64>,
}

impl ReviewTrustKey {
    pub fn new(
        key_id: impl Into<String>,
        public_key_bytes: [u8; 32],
        not_before_unix_s: u64,
        not_after_unix_s: Option<u64>,
        revoked_at_unix_s: Option<u64>,
    ) -> Result<Self, ReviewVerificationError> {
        let key_id = canonical_token(key_id.into(), "key_id", MAX_REVIEW_KEY_ID_BYTES)?;
        if not_after_unix_s.is_some_and(|end| end <= not_before_unix_s) {
            return Err(ReviewVerificationError::InvalidKeyValidityWindow);
        }
        if revoked_at_unix_s.is_some_and(|revoked| revoked < not_before_unix_s) {
            return Err(ReviewVerificationError::InvalidRevocationTime);
        }
        let verifying_key = VerifyingKey::from_bytes(&public_key_bytes)
            .map_err(|_| ReviewVerificationError::InvalidPublicKey)?;
        Ok(Self {
            key_id,
            verifying_key,
            not_before_unix_s,
            not_after_unix_s,
            revoked_at_unix_s,
        })
    }

    pub fn key_id(&self) -> &str {
        &self.key_id
    }

    pub fn public_key_bytes(&self) -> [u8; 32] {
        self.verifying_key.to_bytes()
    }
}

/// Explicit, bounded review-signing trust snapshot.
///
/// The snapshot's delivery and authenticity are intentionally outside this crate.
/// Consumers must only construct this value from an authenticated trust source.
#[derive(Debug, Clone)]
pub struct ReviewTrustStore {
    sequence: u64,
    issued_at_unix_s: u64,
    expires_at_unix_s: u64,
    keys: BTreeMap<String, ReviewTrustKey>,
}

impl ReviewTrustStore {
    pub fn new(
        sequence: u64,
        issued_at_unix_s: u64,
        expires_at_unix_s: u64,
        keys: Vec<ReviewTrustKey>,
    ) -> Result<Self, ReviewVerificationError> {
        if sequence == 0 || issued_at_unix_s >= expires_at_unix_s {
            return Err(ReviewVerificationError::InvalidTrustWindow);
        }
        if keys.is_empty() {
            return Err(ReviewVerificationError::EmptyTrustStore);
        }
        if keys.len() > MAX_REVIEW_TRUST_KEYS {
            return Err(ReviewVerificationError::TooManyTrustKeys {
                actual: keys.len(),
                maximum: MAX_REVIEW_TRUST_KEYS,
            });
        }

        let mut by_id = BTreeMap::new();
        let mut public_keys = BTreeSet::new();
        for key in keys {
            let key_id = key.key_id.clone();
            if by_id.contains_key(&key_id) {
                return Err(ReviewVerificationError::DuplicateTrustKey(key_id));
            }
            if !public_keys.insert(key.public_key_bytes()) {
                return Err(ReviewVerificationError::DuplicateTrustPublicKey(key_id));
            }
            by_id.insert(key_id, key);
        }
        Ok(Self {
            sequence,
            issued_at_unix_s,
            expires_at_unix_s,
            keys: by_id,
        })
    }

    pub fn sequence(&self) -> u64 {
        self.sequence
    }

    /// Content digest for this canonical in-memory trust snapshot.
    ///
    /// A digest helps detect same-sequence substitution, but does not authenticate
    /// the snapshot source or make sequence state durable.
    pub fn canonical_digest(&self) -> String {
        let mut fields = vec![
            REVIEW_TRUST_STORE_SCHEMA.to_owned(),
            format!("sequence={}", self.sequence),
            format!("issued_at_unix_s={}", self.issued_at_unix_s),
            format!("expires_at_unix_s={}", self.expires_at_unix_s),
        ];
        for key in self.keys.values() {
            fields.push(format!("key_id={}", key.key_id));
            fields.push(format!(
                "public_key_hex={}",
                encode_hex(&key.public_key_bytes())
            ));
            fields.push(format!("not_before_unix_s={}", key.not_before_unix_s));
            fields.push(format!(
                "not_after_unix_s={}",
                key.not_after_unix_s
                    .map_or_else(|| "none".to_owned(), |value| value.to_string())
            ));
            fields.push(format!(
                "revoked_at_unix_s={}",
                key.revoked_at_unix_s
                    .map_or_else(|| "none".to_owned(), |value| value.to_string())
            ));
        }
        crate::sha256_hex(format!("{}\n", fields.join("\n")).as_bytes())
    }

    pub fn verify(
        &self,
        receipt: &SignedCalibrationReviewReceipt,
        expected_record_id: &str,
        expected_certificate: &ArtifactReference,
        now_unix_s: u64,
    ) -> Result<(), ReviewVerificationError> {
        if now_unix_s < self.issued_at_unix_s || now_unix_s >= self.expires_at_unix_s {
            return Err(ReviewVerificationError::TrustSnapshotStale);
        }
        if receipt.payload.record_id != expected_record_id {
            return Err(ReviewVerificationError::CalibrationRecordMismatch);
        }
        if receipt.payload.certificate_artifact_id != expected_certificate.artifact_id()
            || receipt.payload.certificate_sha256_hex != expected_certificate.sha256_hex()
        {
            return Err(ReviewVerificationError::CertificateReferenceMismatch);
        }
        if receipt.payload.issued_at_unix_s > now_unix_s {
            return Err(ReviewVerificationError::ReceiptNotYetValid);
        }
        if now_unix_s >= receipt.payload.expires_at_unix_s {
            return Err(ReviewVerificationError::ReceiptExpired);
        }

        let key = self
            .keys
            .get(&receipt.payload.key_id)
            .ok_or_else(|| ReviewVerificationError::UnknownSigner(
                receipt.payload.key_id.clone(),
            ))?;
        if key.revoked_at_unix_s.is_some_and(|revoked| revoked <= now_unix_s) {
            return Err(ReviewVerificationError::SignerRevoked(
                receipt.payload.key_id.clone(),
            ));
        }
        if receipt.payload.issued_at_unix_s < key.not_before_unix_s {
            return Err(ReviewVerificationError::SignerNotYetValid(
                receipt.payload.key_id.clone(),
            ));
        }
        if key
            .not_after_unix_s
            .is_some_and(|end| receipt.payload.issued_at_unix_s >= end)
        {
            return Err(ReviewVerificationError::SignerExpiredAtIssue(
                receipt.payload.key_id.clone(),
            ));
        }

        let signature = Signature::from_bytes(&receipt.signature);
        key.verifying_key
            .verify_strict(&receipt.payload.signing_bytes(), &signature)
            .map_err(|_| ReviewVerificationError::InvalidSignature)
    }
}

/// The signed claims covered by a calibration review signature.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CalibrationReviewPayload {
    key_id: String,
    record_id: String,
    certificate_artifact_id: String,
    certificate_sha256_hex: String,
    issued_at_unix_s: u64,
    expires_at_unix_s: u64,
}

impl CalibrationReviewPayload {
    pub fn new(
        key_id: impl Into<String>,
        record_id: impl Into<String>,
        certificate: &ArtifactReference,
        issued_at_unix_s: u64,
        expires_at_unix_s: u64,
    ) -> Result<Self, ReviewVerificationError> {
        if issued_at_unix_s >= expires_at_unix_s {
            return Err(ReviewVerificationError::InvalidReceiptValidityWindow);
        }
        Ok(Self {
            key_id: canonical_token(key_id.into(), "key_id", MAX_REVIEW_KEY_ID_BYTES)?,
            record_id: canonical_token(
                record_id.into(),
                "record_id",
                MAX_REVIEW_RECORD_ID_BYTES,
            )?,
            certificate_artifact_id: canonical_token(
                certificate.artifact_id().to_owned(),
                "certificate_artifact_id",
                MAX_REVIEW_RECORD_ID_BYTES,
            )?,
            certificate_sha256_hex: canonical_digest(certificate.sha256_hex())?,
            issued_at_unix_s,
            expires_at_unix_s,
        })
    }

    pub fn key_id(&self) -> &str {
        &self.key_id
    }

    pub fn record_id(&self) -> &str {
        &self.record_id
    }

    pub fn certificate_artifact_id(&self) -> &str {
        &self.certificate_artifact_id
    }

    pub fn certificate_sha256_hex(&self) -> &str {
        &self.certificate_sha256_hex
    }

    pub fn issued_at_unix_s(&self) -> u64 {
        self.issued_at_unix_s
    }

    pub fn expires_at_unix_s(&self) -> u64 {
        self.expires_at_unix_s
    }

    /// Canonical domain-separated bytes to sign; independent of receipt encoding.
    pub fn signing_bytes(&self) -> Vec<u8> {
        let fields = [
            CALIBRATION_REVIEW_SCHEMA.to_owned(),
            format!("key_id={}", self.key_id),
            format!("record_id={}", self.record_id),
            format!("certificate_artifact_id={}", self.certificate_artifact_id),
            format!("certificate_sha256={}", self.certificate_sha256_hex),
            format!("issued_at_unix_s={}", self.issued_at_unix_s),
            format!("expires_at_unix_s={}", self.expires_at_unix_s),
        ];
        format!("{}\n", fields.join("\n")).into_bytes()
    }
}

/// Canonical signed calibration review receipt with a detached Ed25519 signature.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SignedCalibrationReviewReceipt {
    payload: CalibrationReviewPayload,
    signature: [u8; 64],
}

impl SignedCalibrationReviewReceipt {
    pub fn new(payload: CalibrationReviewPayload, signature: [u8; 64]) -> Self {
        Self { payload, signature }
    }

    pub fn payload(&self) -> &CalibrationReviewPayload {
        &self.payload
    }

    /// Encode a single canonical envelope. Parsers reject alternate encodings.
    pub fn to_canonical_bytes(&self) -> Vec<u8> {
        let signature_hex = self
            .signature
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        let mut bytes = self.payload.signing_bytes();
        bytes.extend_from_slice(format!("signature_hex={signature_hex}\n").as_bytes());
        bytes
    }

    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, ReviewVerificationError> {
        let text = std::str::from_utf8(bytes)
            .map_err(|_| ReviewVerificationError::InvalidEncoding)?;
        let lines: Vec<&str> = text.lines().collect();
        if lines.len() != 8 || lines[0] != CALIBRATION_REVIEW_SCHEMA {
            return Err(ReviewVerificationError::UnsupportedSchemaOrLayout);
        }
        let key_id = field(lines[1], "key_id")?;
        let record_id = field(lines[2], "record_id")?;
        let certificate_artifact_id = field(lines[3], "certificate_artifact_id")?;
        let certificate_sha256_hex = field(lines[4], "certificate_sha256")?;
        let issued_at_unix_s = field(lines[5], "issued_at_unix_s")?
            .parse::<u64>()
            .map_err(|_| ReviewVerificationError::InvalidTimestamp)?;
        let expires_at_unix_s = field(lines[6], "expires_at_unix_s")?
            .parse::<u64>()
            .map_err(|_| ReviewVerificationError::InvalidTimestamp)?;
        let signature_hex = field(lines[7], "signature_hex")?;
        let signature = decode_signature(signature_hex)?;
        let certificate =
            ArtifactReference::new(certificate_artifact_id, certificate_sha256_hex)
                .map_err(|_| ReviewVerificationError::InvalidCertificateReference)?;
        let payload = CalibrationReviewPayload::new(
            key_id,
            record_id,
            &certificate,
            issued_at_unix_s,
            expires_at_unix_s,
        )?;
        let receipt = Self { payload, signature };
        if receipt.to_canonical_bytes() != bytes {
            return Err(ReviewVerificationError::NonCanonicalEncoding);
        }
        Ok(receipt)
    }
}

/// Tracks monotonic trust-store sequence/digest admission within one process.
///
/// Persist and authenticate the last sequence and digest across restarts when
/// rollback resistance is required. A freshly constructed default guard has no
/// history and therefore cannot detect rollback to a previously seen snapshot.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ReviewTrustStoreGuard {
    last_sequence: Option<u64>,
    last_digest: Option<String>,
}

impl ReviewTrustStoreGuard {
    pub fn last_sequence(&self) -> Option<u64> {
        self.last_sequence
    }

    /// Admit a fresh snapshot if it advances state or exactly repeats the latest one.
    pub fn admit(
        &mut self,
        trust_store: &ReviewTrustStore,
        now_unix_s: u64,
    ) -> Result<(), ReviewVerificationError> {
        if now_unix_s < trust_store.issued_at_unix_s
            || now_unix_s >= trust_store.expires_at_unix_s
        {
            return Err(ReviewVerificationError::TrustSnapshotStale);
        }
        let digest = trust_store.canonical_digest();
        if let Some(highest) = self.last_sequence {
            if trust_store.sequence < highest {
                return Err(ReviewVerificationError::TrustSnapshotRollback {
                    highest_sequence: highest,
                    received_sequence: trust_store.sequence,
                });
            }
            if trust_store.sequence == highest {
                if self.last_digest.as_deref() != Some(digest.as_str()) {
                    return Err(ReviewVerificationError::TrustSnapshotSequenceReused {
                        sequence: highest,
                    });
                }
                return Ok(());
            }
        }
        self.last_sequence = Some(trust_store.sequence);
        self.last_digest = Some(digest);
        Ok(())
    }
}

/// A receipt which passed byte-digest, certificate-binding, lifecycle, freshness,
/// and strict Ed25519 checks against the supplied trust store.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedCalibrationReviewReceipt {
    artifact: VerifiedArtifactBytes,
    receipt: SignedCalibrationReviewReceipt,
    verified_at_unix_s: u64,
}

impl VerifiedCalibrationReviewReceipt {
    pub fn verify(
        artifact: VerifiedArtifactBytes,
        trust_store: &ReviewTrustStore,
        trust_store_guard: &mut ReviewTrustStoreGuard,
        expected_record_id: &str,
        expected_certificate: &ArtifactReference,
        now_unix_s: u64,
    ) -> Result<Self, ReviewVerificationError> {
        if artifact.byte_length() == 0 {
            return Err(ReviewVerificationError::EmptyReceipt);
        }
        let receipt = SignedCalibrationReviewReceipt::from_canonical_bytes(artifact.bytes())?;
        trust_store_guard.admit(trust_store, now_unix_s)?;
        trust_store.verify(
            &receipt,
            expected_record_id,
            expected_certificate,
            now_unix_s,
        )?;
        Ok(Self {
            artifact,
            receipt,
            verified_at_unix_s: now_unix_s,
        })
    }

    pub fn artifact(&self) -> &VerifiedArtifactBytes {
        &self.artifact
    }

    pub fn payload(&self) -> &CalibrationReviewPayload {
        self.receipt.payload()
    }

    pub fn verified_at_unix_s(&self) -> u64 {
        self.verified_at_unix_s
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReviewVerificationError {
    InvalidField(&'static str),
    InvalidCertificateReference,
    InvalidDigest,
    InvalidEncoding,
    UnsupportedSchemaOrLayout,
    NonCanonicalEncoding,
    InvalidTimestamp,
    InvalidSignatureEncoding,
    InvalidSignature,
    InvalidPublicKey,
    InvalidKeyValidityWindow,
    InvalidRevocationTime,
    InvalidTrustWindow,
    EmptyTrustStore,
    TooManyTrustKeys { actual: usize, maximum: usize },
    DuplicateTrustKey(String),
    DuplicateTrustPublicKey(String),
    TrustSnapshotRollback {
        highest_sequence: u64,
        received_sequence: u64,
    },
    TrustSnapshotSequenceReused {
        sequence: u64,
    },
    TrustSnapshotStale,
    UnknownSigner(String),
    SignerNotYetValid(String),
    SignerExpiredAtIssue(String),
    SignerRevoked(String),
    InvalidReceiptValidityWindow,
    ReceiptNotYetValid,
    ReceiptExpired,
    CalibrationRecordMismatch,
    CertificateReferenceMismatch,
    EmptyReceipt,
}

impl fmt::Display for ReviewVerificationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl std::error::Error for ReviewVerificationError {}

fn canonical_token(
    value: String,
    field_name: &'static str,
    maximum_bytes: usize,
) -> Result<String, ReviewVerificationError> {
    if value.is_empty()
        || value.trim() != value
        || value.len() > maximum_bytes
        || value.bytes().any(|byte| byte.is_ascii_control())
    {
        return Err(ReviewVerificationError::InvalidField(field_name));
    }
    Ok(value)
}

fn encode_hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn canonical_digest(value: &str) -> Result<String, ReviewVerificationError> {
    if value.len() != 64
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(ReviewVerificationError::InvalidDigest);
    }
    Ok(value.to_owned())
}

fn field<'a>(
    line: &'a str,
    prefix: &'static str,
) -> Result<&'a str, ReviewVerificationError> {
    line.strip_prefix(&format!("{prefix}="))
        .ok_or(ReviewVerificationError::UnsupportedSchemaOrLayout)
}

fn decode_signature(value: &str) -> Result<[u8; 64], ReviewVerificationError> {
    if value.len() != 128
        || !value
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err(ReviewVerificationError::InvalidSignatureEncoding);
    }
    let mut signature = [0_u8; 64];
    for (index, byte) in signature.iter_mut().enumerate() {
        *byte = u8::from_str_radix(&value[index * 2..index * 2 + 2], 16)
            .map_err(|_| ReviewVerificationError::InvalidSignatureEncoding)?;
    }
    Ok(signature)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};

    const NOW: u64 = 1_750_000_000;
    const TRUST_SEQUENCE: u64 = 4;

    fn signing_key() -> SigningKey {
        SigningKey::from_bytes(&[0x42; 32])
    }

    fn certificate() -> ArtifactReference {
        ArtifactReference::new(
            "calibration-certificate-1",
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        )
        .unwrap()
    }

    fn trust_store() -> ReviewTrustStore {
        ReviewTrustStore::new(
            TRUST_SEQUENCE,
            NOW - 10,
            NOW + 100,
            vec![
                ReviewTrustKey::new(
                    "reviewer-a",
                    signing_key().verifying_key().to_bytes(),
                    NOW - 100,
                    Some(NOW + 500),
                    None,
                )
                .unwrap(),
            ],
        )
        .unwrap()
    }

    fn signed_receipt() -> SignedCalibrationReviewReceipt {
        let payload = CalibrationReviewPayload::new(
            "reviewer-a",
            "calibration-record-1",
            &certificate(),
            NOW - 1,
            NOW + 50,
        )
        .unwrap();
        let signature = signing_key().sign(&payload.signing_bytes()).to_bytes();
        SignedCalibrationReviewReceipt::new(payload, signature)
    }

    fn verified_artifact(
        receipt: &SignedCalibrationReviewReceipt,
    ) -> VerifiedArtifactBytes {
        let bytes = receipt.to_canonical_bytes();
        let digest = crate::sha256_hex(&bytes);
        VerifiedArtifactBytes::verify(
            ArtifactReference::new("review-receipt-1", digest).unwrap(),
            bytes,
        )
        .unwrap()
    }

    #[test]
    fn canonical_receipt_roundtrips_and_verifies_with_the_configured_trust_key() {
        let receipt = signed_receipt();
        let bytes = receipt.to_canonical_bytes();
        let parsed = SignedCalibrationReviewReceipt::from_canonical_bytes(&bytes).unwrap();
        assert_eq!(parsed, receipt);

        let trust = trust_store();
        let mut trust_guard = ReviewTrustStoreGuard::default();
        let verified = VerifiedCalibrationReviewReceipt::verify(
            verified_artifact(&receipt),
            &trust,
            &mut trust_guard,
            "calibration-record-1",
            &certificate(),
            NOW,
        )
        .unwrap();
        assert_eq!(verified.payload().key_id(), "reviewer-a");
        assert_eq!(verified.verified_at_unix_s(), NOW);
        assert_eq!(trust.sequence(), TRUST_SEQUENCE);
        assert_eq!(trust_guard.last_sequence(), Some(TRUST_SEQUENCE));
    }

    #[test]
    fn signature_tampering_is_rejected_even_when_the_artifact_digest_is_updated() {
        let mut receipt = signed_receipt();
        let mut signature = receipt.signature;
        signature[0] ^= 1;
        receipt.signature = signature;
        let artifact = verified_artifact(&receipt);

        let trust = trust_store();
        let mut trust_guard = ReviewTrustStoreGuard::default();
        assert_eq!(
            VerifiedCalibrationReviewReceipt::verify(
                artifact,
                &trust,
                &mut trust_guard,
                "calibration-record-1",
                &certificate(),
                NOW,
            )
            .unwrap_err(),
            ReviewVerificationError::InvalidSignature
        );
    }

    #[test]
    fn unknown_revoked_and_expired_signers_fail_closed() {
        let receipt = signed_receipt();
        let unknown = ReviewTrustStore::new(
            TRUST_SEQUENCE,
            NOW - 10,
            NOW + 100,
            vec![ReviewTrustKey::new(
                "different-key",
                signing_key().verifying_key().to_bytes(),
                NOW - 100,
                Some(NOW + 500),
                None,
            )
            .unwrap()],
        )
        .unwrap();
        assert!(matches!(
            unknown.verify(&receipt, "calibration-record-1", &certificate(), NOW),
            Err(ReviewVerificationError::UnknownSigner(_))
        ));

        let revoked = ReviewTrustStore::new(
            TRUST_SEQUENCE,
            NOW - 10,
            NOW + 100,
            vec![ReviewTrustKey::new(
                "reviewer-a",
                signing_key().verifying_key().to_bytes(),
                NOW - 100,
                Some(NOW + 500),
                Some(NOW),
            )
            .unwrap()],
        )
        .unwrap();
        assert_eq!(
            revoked.verify(&receipt, "calibration-record-1", &certificate(), NOW),
            Err(ReviewVerificationError::SignerRevoked("reviewer-a".into()))
        );

        let expired = ReviewTrustStore::new(
            TRUST_SEQUENCE,
            NOW - 10,
            NOW + 100,
            vec![
                ReviewTrustKey::new(
                    "reviewer-a",
                    signing_key().verifying_key().to_bytes(),
                    NOW - 100,
                    Some(NOW - 2),
                    None,
                )
                .unwrap(),
            ],
        )
        .unwrap();
        assert_eq!(
            expired.verify(&receipt, "calibration-record-1", &certificate(), NOW),
            Err(ReviewVerificationError::SignerExpiredAtIssue(
                "reviewer-a".into()
            ))
        );
    }

    #[test]
    fn trust_store_rejects_empty_and_duplicate_key_material() {
        assert_eq!(
            ReviewTrustStore::new(TRUST_SEQUENCE, NOW - 10, NOW + 100, Vec::new())
                .unwrap_err(),
            ReviewVerificationError::EmptyTrustStore
        );
        assert_eq!(
            ReviewTrustStore::new(0, NOW - 10, NOW + 100, vec![
                ReviewTrustKey::new(
                    "reviewer-a",
                    signing_key().verifying_key().to_bytes(),
                    NOW - 100,
                    Some(NOW + 500),
                    None,
                )
                .unwrap(),
            ])
            .unwrap_err(),
            ReviewVerificationError::InvalidTrustWindow
        );
        assert_eq!(
            ReviewTrustStore::new(
                TRUST_SEQUENCE,
                NOW - 10,
                NOW + 100,
                vec![
                    ReviewTrustKey::new(
                        "reviewer-a",
                        signing_key().verifying_key().to_bytes(),
                        NOW - 100,
                        Some(NOW + 500),
                        None,
                    )
                    .unwrap(),
                    ReviewTrustKey::new(
                        "reviewer-alias",
                        signing_key().verifying_key().to_bytes(),
                        NOW - 100,
                        Some(NOW + 500),
                        None,
                    )
                    .unwrap(),
                ],
            )
            .unwrap_err(),
            ReviewVerificationError::DuplicateTrustPublicKey("reviewer-alias".into())
        );
    }

    #[test]
    fn trust_store_guard_allows_idempotent_rechecks_and_monotonic_advance() {
        let initial = trust_store();
        let advanced = ReviewTrustStore::new(
            TRUST_SEQUENCE + 1,
            NOW - 10,
            NOW + 200,
            vec![
                ReviewTrustKey::new(
                    "reviewer-a",
                    signing_key().verifying_key().to_bytes(),
                    NOW - 100,
                    Some(NOW + 500),
                    None,
                )
                .unwrap(),
            ],
        )
        .unwrap();
        let mut guard = ReviewTrustStoreGuard::default();

        guard.admit(&initial, NOW).unwrap();
        guard.admit(&initial, NOW).unwrap();
        guard.admit(&advanced, NOW).unwrap();
        assert_eq!(guard.last_sequence(), Some(TRUST_SEQUENCE + 1));
    }

    #[test]
    fn trust_store_guard_rejects_sequence_rollback() {
        let newer = ReviewTrustStore::new(
            TRUST_SEQUENCE + 1,
            NOW - 10,
            NOW + 200,
            vec![
                ReviewTrustKey::new(
                    "reviewer-a",
                    signing_key().verifying_key().to_bytes(),
                    NOW - 100,
                    Some(NOW + 500),
                    None,
                )
                .unwrap(),
            ],
        )
        .unwrap();
        let mut guard = ReviewTrustStoreGuard::default();
        guard.admit(&newer, NOW).unwrap();

        assert_eq!(
            guard.admit(&trust_store(), NOW),
            Err(ReviewVerificationError::TrustSnapshotRollback {
                highest_sequence: TRUST_SEQUENCE + 1,
                received_sequence: TRUST_SEQUENCE,
            })
        );
    }

    #[test]
    fn trust_store_guard_rejects_different_contents_at_the_same_sequence() {
        let replacement = ReviewTrustStore::new(
            TRUST_SEQUENCE,
            NOW - 10,
            NOW + 100,
            vec![
                ReviewTrustKey::new(
                    "reviewer-b",
                    SigningKey::from_bytes(&[0x43; 32])
                        .verifying_key()
                        .to_bytes(),
                    NOW - 100,
                    Some(NOW + 500),
                    None,
                )
                .unwrap(),
            ],
        )
        .unwrap();
        let mut guard = ReviewTrustStoreGuard::default();
        guard.admit(&trust_store(), NOW).unwrap();

        assert_eq!(
            guard.admit(&replacement, NOW),
            Err(ReviewVerificationError::TrustSnapshotSequenceReused {
                sequence: TRUST_SEQUENCE,
            })
        );
    }

    #[test]
    fn trust_snapshot_and_receipt_expiration_are_enforced() {
        let receipt = signed_receipt();
        assert_eq!(
            trust_store().verify(&receipt, "calibration-record-1", &certificate(), NOW + 100),
            Err(ReviewVerificationError::TrustSnapshotStale)
        );
        assert_eq!(
            trust_store().verify(
                &receipt,
                "calibration-record-1",
                &certificate(),
                NOW + 50,
            ),
            Err(ReviewVerificationError::ReceiptExpired)
        );
    }

    #[test]
    fn receipt_must_bind_the_expected_certificate_and_record() {
        let receipt = signed_receipt();
        assert_eq!(
            trust_store().verify(&receipt, "another-record", &certificate(), NOW),
            Err(ReviewVerificationError::CalibrationRecordMismatch)
        );

        let other_certificate = ArtifactReference::new(
            "calibration-certificate-other",
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        )
        .unwrap();
        assert_eq!(
            trust_store().verify(
                &receipt,
                "calibration-record-1",
                &other_certificate,
                NOW,
            ),
            Err(ReviewVerificationError::CertificateReferenceMismatch)
        );
    }

    #[test]
    fn empty_review_artifact_is_rejected_before_signature_verification() {
        let digest = crate::sha256_hex(b"");
        let artifact = VerifiedArtifactBytes::verify(
            ArtifactReference::new("empty-review", digest).unwrap(),
            Vec::new(),
        )
        .unwrap();
        assert_eq!(
            VerifiedCalibrationReviewReceipt::verify(
                artifact,
                &trust_store(),
                &mut ReviewTrustStoreGuard::default(),
                "calibration-record-1",
                &certificate(),
                NOW,
            )
            .unwrap_err(),
            ReviewVerificationError::EmptyReceipt
        );
    }

    #[test]
    fn alternate_noncanonical_encoding_is_rejected() {
        let receipt = signed_receipt();
        let mut bytes = receipt.to_canonical_bytes();
        bytes.extend_from_slice(b"extra=field\n");
        assert_eq!(
            SignedCalibrationReviewReceipt::from_canonical_bytes(&bytes).unwrap_err(),
            ReviewVerificationError::UnsupportedSchemaOrLayout
        );
    }
}
