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
pub const REVIEW_TRUST_STORE_SCHEMA: &str = "symthaea.calibration-review-trust.v2";
pub const MAX_REVIEW_KEY_ID_BYTES: usize = 256;
pub const MAX_REVIEWER_ID_BYTES: usize = 256;
pub const MAX_REVIEW_ORGANIZATION_ID_BYTES: usize = 256;
pub const MAX_REVIEW_EXCLUDED_ORGANIZATIONS: usize = 4096;
pub const MAX_REVIEW_RECORD_ID_BYTES: usize = 256;
pub const MAX_REVIEW_TRUST_KEYS: usize = 4096;

/// Review authority input with an explicit public key lifecycle.
#[derive(Debug, Clone)]
pub struct ReviewTrustKey {
    key_id: String,
    verifying_key: VerifyingKey,
    reviewer_id: Option<String>,
    organization_id: Option<String>,
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
            reviewer_id: None,
            organization_id: None,
            not_before_unix_s,
            not_after_unix_s,
            revoked_at_unix_s,
        })
    }

    /// Bind this signing key to a reviewer identity and organization.
    ///
    /// These claims are covered by the root-signed trust snapshot. A key without
    /// both fields can verify a generic signature but cannot approve calibration evidence.
    pub fn with_reviewer_authorization(
        mut self,
        reviewer_id: impl Into<String>,
        organization_id: impl Into<String>,
    ) -> Result<Self, ReviewVerificationError> {
        self.reviewer_id = Some(canonical_token(
            reviewer_id.into(),
            "reviewer_id",
            MAX_REVIEWER_ID_BYTES,
        )?);
        self.organization_id = Some(canonical_token(
            organization_id.into(),
            "organization_id",
            MAX_REVIEW_ORGANIZATION_ID_BYTES,
        )?);
        Ok(self)
    }

    pub fn key_id(&self) -> &str {
        &self.key_id
    }

    pub fn reviewer_id(&self) -> Option<&str> {
        self.reviewer_id.as_deref()
    }

    pub fn organization_id(&self) -> Option<&str> {
        self.organization_id.as_deref()
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

    /// Canonical, deterministic binary encoding of this trust snapshot.
    ///
    /// Keys are emitted in their canonical key-ID order. The schema tag, all
    /// validity/revocation windows, public keys, and sequence are signed together.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::new();
        bytes.extend_from_slice(b"SYMRTRS2\0");
        append_len_prefixed(&mut bytes, REVIEW_TRUST_STORE_SCHEMA.as_bytes());
        bytes.extend_from_slice(&self.sequence.to_le_bytes());
        bytes.extend_from_slice(&self.issued_at_unix_s.to_le_bytes());
        bytes.extend_from_slice(&self.expires_at_unix_s.to_le_bytes());
        bytes.extend_from_slice(&(self.keys.len() as u32).to_le_bytes());
        for key in self.keys.values() {
            append_len_prefixed(&mut bytes, key.key_id.as_bytes());
            append_optional_token(&mut bytes, key.reviewer_id.as_deref());
            append_optional_token(&mut bytes, key.organization_id.as_deref());
            bytes.extend_from_slice(&key.public_key_bytes());
            bytes.extend_from_slice(&key.not_before_unix_s.to_le_bytes());
            append_optional_u64(&mut bytes, key.not_after_unix_s);
            append_optional_u64(&mut bytes, key.revoked_at_unix_s);
        }
        bytes
    }

    /// Decode and enforce the canonical binary snapshot representation.
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, ReviewVerificationError> {
        const MAGIC: &[u8] = b"SYMRTRS2\0";
        if !bytes.starts_with(MAGIC) {
            return Err(ReviewVerificationError::UnsupportedSchemaOrLayout);
        }
        let mut offset = MAGIC.len();
        let schema = read_len_prefixed(bytes, &mut offset)?;
        if schema != REVIEW_TRUST_STORE_SCHEMA.as_bytes() {
            return Err(ReviewVerificationError::UnsupportedSchemaOrLayout);
        }
        let sequence = read_u64(bytes, &mut offset)?;
        let issued_at_unix_s = read_u64(bytes, &mut offset)?;
        let expires_at_unix_s = read_u64(bytes, &mut offset)?;
        let key_count = read_u32(bytes, &mut offset)? as usize;
        if key_count > MAX_REVIEW_TRUST_KEYS {
            return Err(ReviewVerificationError::TooManyTrustKeys {
                actual: key_count,
                maximum: MAX_REVIEW_TRUST_KEYS,
            });
        }
        let mut keys = Vec::with_capacity(key_count);
        for _ in 0..key_count {
            let key_id = std::str::from_utf8(read_len_prefixed(bytes, &mut offset)?)
                .map_err(|_| ReviewVerificationError::InvalidEncoding)?
                .to_owned();
            let reviewer_id = read_optional_token(
                bytes,
                &mut offset,
                "reviewer_id",
                MAX_REVIEWER_ID_BYTES,
            )?;
            let organization_id = read_optional_token(
                bytes,
                &mut offset,
                "organization_id",
                MAX_REVIEW_ORGANIZATION_ID_BYTES,
            )?;
            let public_key_bytes = read_fixed::<32>(bytes, &mut offset)?;
            let not_before_unix_s = read_u64(bytes, &mut offset)?;
            let not_after_unix_s = read_optional_u64(bytes, &mut offset)?;
            let revoked_at_unix_s = read_optional_u64(bytes, &mut offset)?;
            let mut key = ReviewTrustKey::new(
                key_id,
                public_key_bytes,
                not_before_unix_s,
                not_after_unix_s,
                revoked_at_unix_s,
            )?;
            match (reviewer_id, organization_id) {
                (Some(reviewer_id), Some(organization_id)) => {
                    key = key.with_reviewer_authorization(reviewer_id, organization_id)?;
                }
                (None, None) => {}
                _ => return Err(ReviewVerificationError::InvalidEncoding),
            }
            keys.push(key);
        }
        if offset != bytes.len() {
            return Err(ReviewVerificationError::NonCanonicalEncoding);
        }
        let snapshot = Self::new(sequence, issued_at_unix_s, expires_at_unix_s, keys)?;
        if snapshot.canonical_bytes() != bytes {
            return Err(ReviewVerificationError::NonCanonicalEncoding);
        }
        Ok(snapshot)
    }

    /// Digest the canonical trust contents (not the authority signature).
    pub fn canonical_digest(&self) -> String {
        crate::sha256_hex(&self.canonical_bytes())
    }

    fn authorize_reviewer(
        &self,
        key_id: &str,
        policy: &ReviewAuthorizationPolicy,
        trust_store_sequence: u64,
        trust_store_sha256: String,
        trust_authority_key_sha256: String,
        verified_at_unix_s: u64,
    ) -> Result<VerifiedReviewerAuthorization, ReviewVerificationError> {
        let key = self
            .keys
            .get(key_id)
            .ok_or_else(|| ReviewVerificationError::UnknownSigner(key_id.to_owned()))?;
        let reviewer_id = key
            .reviewer_id
            .as_deref()
            .ok_or_else(|| ReviewVerificationError::ReviewerNotAuthorized(key_id.to_owned()))?;
        let reviewer_organization_id = key.organization_id.as_deref().ok_or_else(|| {
            ReviewVerificationError::ReviewerNotAuthorized(key_id.to_owned())
        })?;

        if reviewer_organization_id == policy.provider_organization_id {
            return Err(ReviewVerificationError::ReviewerNotIndependent {
                reviewer_organization_id: reviewer_organization_id.to_owned(),
                conflicted_organization_id: policy.provider_organization_id.clone(),
            });
        }
        if policy.excluded_organization_ids.contains(reviewer_organization_id) {
            return Err(ReviewVerificationError::ReviewerNotIndependent {
                reviewer_organization_id: reviewer_organization_id.to_owned(),
                conflicted_organization_id: reviewer_organization_id.to_owned(),
            });
        }

        Ok(VerifiedReviewerAuthorization {
            reviewer_key_id: key_id.to_owned(),
            reviewer_id: reviewer_id.to_owned(),
            reviewer_organization_id: reviewer_organization_id.to_owned(),
            provider_organization_id: policy.provider_organization_id.clone(),
            excluded_organization_ids: policy.excluded_organization_ids.iter().cloned().collect(),
            policy_sha256: policy.canonical_digest(),
            trust_store_sequence,
            trust_store_sha256,
            trust_authority_key_sha256,
            verified_at_unix_s,
        })
    }

    fn verify(
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

/// Signed canonical trust-store snapshot.
#[derive(Debug, Clone)]
pub struct SignedReviewTrustStoreSnapshot {
    trust_store: ReviewTrustStore,
    signature: [u8; 64],
}

impl SignedReviewTrustStoreSnapshot {
    pub fn new(trust_store: ReviewTrustStore, signature: [u8; 64]) -> Self {
        Self {
            trust_store,
            signature,
        }
    }

    /// Canonical domain-separated bytes an authority signs out of process.
    pub fn signing_message(trust_store: &ReviewTrustStore) -> Vec<u8> {
        let mut message = b"symthaea.calibration-review-trust-signature.v2\0".to_vec();
        message.extend_from_slice(&trust_store.canonical_bytes());
        message
    }

    /// Encode the signed snapshot with a fixed magic, canonical store body and signature.
    pub fn to_canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = b"SYMRTRSS2\0".to_vec();
        bytes.extend_from_slice(&self.trust_store.canonical_bytes());
        bytes.extend_from_slice(&self.signature);
        bytes
    }

    /// Parse a signed snapshot and reject alternate/trailing encodings.
    pub fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, ReviewVerificationError> {
        const MAGIC: &[u8] = b"SYMRTRSS2\0";
        if !bytes.starts_with(MAGIC) || bytes.len() < MAGIC.len() + 64 {
            return Err(ReviewVerificationError::UnsupportedSchemaOrLayout);
        }
        let signature_start = bytes.len() - 64;
        if signature_start <= MAGIC.len() {
            return Err(ReviewVerificationError::InvalidEncoding);
        }
        let store = ReviewTrustStore::from_canonical_bytes(
            &bytes[MAGIC.len()..signature_start],
        )?;
        let signature: [u8; 64] = bytes[signature_start..]
            .try_into()
            .map_err(|_| ReviewVerificationError::InvalidSignatureEncoding)?;
        let snapshot = Self::new(store, signature);
        if snapshot.to_canonical_bytes() != bytes {
            return Err(ReviewVerificationError::NonCanonicalEncoding);
        }
        Ok(snapshot)
    }

    /// Authenticate the snapshot against an authority key pinned by the caller.
    pub fn verify(
        self,
        authority_key: &VerifyingKey,
    ) -> Result<VerifiedReviewTrustStore, ReviewVerificationError> {
        let signature = Signature::from_bytes(&self.signature);
        authority_key
            .verify_strict(&Self::signing_message(&self.trust_store), &signature)
            .map_err(|_| ReviewVerificationError::InvalidTrustSnapshotSignature)?;
        Ok(VerifiedReviewTrustStore {
            trust_store: self.trust_store,
            authority_key_sha256: crate::sha256_hex(authority_key.as_bytes()),
        })
    }
}

/// Authenticated trust snapshot. Its private fields ensure receipt verification
/// cannot be passed a mutable-looking snapshot that has not passed root-key
/// signature validation.
#[derive(Debug, Clone)]
pub struct VerifiedReviewTrustStore {
    trust_store: ReviewTrustStore,
    authority_key_sha256: String,
}

impl VerifiedReviewTrustStore {
    pub fn trust_store(&self) -> &ReviewTrustStore {
        &self.trust_store
    }

    pub fn sequence(&self) -> u64 {
        self.trust_store.sequence()
    }

    pub fn authority_key_sha256(&self) -> &str {
        &self.authority_key_sha256
    }

    /// Bind both the snapshot contents and the authority that authenticated them.
    pub fn canonical_digest(&self) -> String {
        let mut bytes =
            b"symthaea.calibration-review-verified-trust.v2\0".to_vec();
        bytes.extend_from_slice(&self.trust_store.canonical_bytes());
        bytes.extend_from_slice(self.authority_key_sha256.as_bytes());
        crate::sha256_hex(&bytes)
    }
}

/// Conflict policy for reviewer authorization for a specific provider.
///
/// Provider and exclusion identifiers must come from a separately governed registry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReviewAuthorizationPolicy {
    provider_organization_id: String,
    excluded_organization_ids: BTreeSet<String>,
}

impl ReviewAuthorizationPolicy {
    pub fn new(
        provider_organization_id: impl Into<String>,
        excluded_organization_ids: Vec<String>,
    ) -> Result<Self, ReviewVerificationError> {
        if excluded_organization_ids.len() > MAX_REVIEW_EXCLUDED_ORGANIZATIONS {
            return Err(ReviewVerificationError::TooManyExcludedOrganizations {
                actual: excluded_organization_ids.len(),
                maximum: MAX_REVIEW_EXCLUDED_ORGANIZATIONS,
            });
        }
        let provider_organization_id = canonical_token(
            provider_organization_id.into(),
            "provider_organization_id",
            MAX_REVIEW_ORGANIZATION_ID_BYTES,
        )?;
        let excluded_organization_ids = excluded_organization_ids
            .into_iter()
            .map(|organization_id| {
                canonical_token(
                    organization_id,
                    "excluded_organization_id",
                    MAX_REVIEW_ORGANIZATION_ID_BYTES,
                )
            })
            .collect::<Result<BTreeSet<_>, _>>()?;
        Ok(Self {
            provider_organization_id,
            excluded_organization_ids,
        })
    }

    pub fn provider_organization_id(&self) -> &str {
        &self.provider_organization_id
    }

    pub fn excluded_organization_ids(&self) -> impl Iterator<Item = &str> {
        self.excluded_organization_ids.iter().map(String::as_str)
    }

    fn canonical_digest(&self) -> String {
        let mut bytes =
            b"symthaea.calibration-review-authorization-policy.v1\0".to_vec();
        append_len_prefixed(&mut bytes, self.provider_organization_id.as_bytes());
        bytes.extend_from_slice(&(self.excluded_organization_ids.len() as u32).to_le_bytes());
        for organization_id in &self.excluded_organization_ids {
            append_len_prefixed(&mut bytes, organization_id.as_bytes());
        }
        crate::sha256_hex(&bytes)
    }
}

/// The root-authenticated reviewer identity plus the applied conflict policy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedReviewerAuthorization {
    reviewer_key_id: String,
    reviewer_id: String,
    reviewer_organization_id: String,
    provider_organization_id: String,
    excluded_organization_ids: Vec<String>,
    policy_sha256: String,
    trust_store_sequence: u64,
    trust_store_sha256: String,
    trust_authority_key_sha256: String,
    verified_at_unix_s: u64,
}

impl VerifiedReviewerAuthorization {
    pub fn reviewer_key_id(&self) -> &str {
        &self.reviewer_key_id
    }

    pub fn reviewer_id(&self) -> &str {
        &self.reviewer_id
    }

    pub fn reviewer_organization_id(&self) -> &str {
        &self.reviewer_organization_id
    }

    pub fn provider_organization_id(&self) -> &str {
        &self.provider_organization_id
    }

    /// Organizations excluded by the exact policy applied during verification.
    pub fn excluded_organization_ids(&self) -> impl Iterator<Item = &str> {
        self.excluded_organization_ids.iter().map(String::as_str)
    }

    pub fn policy_sha256(&self) -> &str {
        &self.policy_sha256
    }

    pub fn trust_store_sequence(&self) -> u64 {
        self.trust_store_sequence
    }

    pub fn trust_store_sha256(&self) -> &str {
        &self.trust_store_sha256
    }

    pub fn trust_authority_key_sha256(&self) -> &str {
        &self.trust_authority_key_sha256
    }

    pub fn verified_at_unix_s(&self) -> u64 {
        self.verified_at_unix_s
    }
}

fn append_len_prefixed(bytes: &mut Vec<u8>, value: &[u8]) {
    // Schema strings and key IDs are bounded by constants before reaching here.
    bytes.extend_from_slice(&(value.len() as u16).to_le_bytes());
    bytes.extend_from_slice(value);
}

fn read_fixed<const N: usize>(
    bytes: &[u8],
    offset: &mut usize,
) -> Result<[u8; N], ReviewVerificationError> {
    let end = (*offset)
        .checked_add(N)
        .ok_or(ReviewVerificationError::InvalidEncoding)?;
    let value = bytes
        .get(*offset..end)
        .ok_or(ReviewVerificationError::InvalidEncoding)?;
    let mut result = [0_u8; N];
    result.copy_from_slice(value);
    *offset = end;
    Ok(result)
}

fn read_len_prefixed<'a>(
    bytes: &'a [u8],
    offset: &mut usize,
) -> Result<&'a [u8], ReviewVerificationError> {
    let len = u16::from_le_bytes(read_fixed::<2>(bytes, offset)?) as usize;
    let end = (*offset)
        .checked_add(len)
        .ok_or(ReviewVerificationError::InvalidEncoding)?;
    let value = bytes
        .get(*offset..end)
        .ok_or(ReviewVerificationError::InvalidEncoding)?;
    *offset = end;
    Ok(value)
}

fn read_u32(bytes: &[u8], offset: &mut usize) -> Result<u32, ReviewVerificationError> {
    Ok(u32::from_le_bytes(read_fixed::<4>(bytes, offset)?))
}

fn read_u64(bytes: &[u8], offset: &mut usize) -> Result<u64, ReviewVerificationError> {
    Ok(u64::from_le_bytes(read_fixed::<8>(bytes, offset)?))
}

fn read_optional_u64(
    bytes: &[u8],
    offset: &mut usize,
) -> Result<Option<u64>, ReviewVerificationError> {
    match read_fixed::<1>(bytes, offset)?[0] {
        0 => Ok(None),
        1 => Ok(Some(read_u64(bytes, offset)?)),
        _ => Err(ReviewVerificationError::InvalidEncoding),
    }
}

fn append_optional_token(bytes: &mut Vec<u8>, value: Option<&str>) {
    match value {
        Some(value) => {
            bytes.push(1);
            append_len_prefixed(bytes, value.as_bytes());
        }
        None => bytes.push(0),
    }
}

fn read_optional_token(
    bytes: &[u8],
    offset: &mut usize,
    field_name: &'static str,
    maximum_bytes: usize,
) -> Result<Option<String>, ReviewVerificationError> {
    match read_fixed::<1>(bytes, offset)?[0] {
        0 => Ok(None),
        1 => {
            let value = std::str::from_utf8(read_len_prefixed(bytes, offset)?)
                .map_err(|_| ReviewVerificationError::InvalidEncoding)?;
            Ok(Some(canonical_token(
                value.to_owned(),
                field_name,
                maximum_bytes,
            )?))
        }
        _ => Err(ReviewVerificationError::InvalidEncoding),
    }
}

fn append_optional_u64(bytes: &mut Vec<u8>, value: Option<u64>) {
    match value {
        Some(value) => {
            bytes.push(1);
            bytes.extend_from_slice(&value.to_le_bytes());
        }
        None => bytes.push(0),
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

    pub fn last_digest(&self) -> Option<&str> {
        self.last_digest.as_deref()
    }

    /// Export state for authenticated durable persistence.
    pub fn checkpoint(&self) -> Option<(u64, String)> {
        self.last_sequence
            .zip(self.last_digest.clone())
    }

    /// Restore a checkpoint only after the caller authenticates its persisted source.
    pub fn from_checkpoint(
        highest_sequence: u64,
        snapshot_digest: impl Into<String>,
    ) -> Result<Self, ReviewVerificationError> {
        let snapshot_digest = canonical_digest(&snapshot_digest.into())?;
        if highest_sequence == 0 {
            return Err(ReviewVerificationError::InvalidTrustCheckpoint);
        }
        Ok(Self {
            last_sequence: Some(highest_sequence),
            last_digest: Some(snapshot_digest),
        })
    }

    /// Validate a candidate without changing the high-water mark.
    fn check_candidate(
        &self,
        trust_store: &VerifiedReviewTrustStore,
        now_unix_s: u64,
    ) -> Result<(), ReviewVerificationError> {
        let store = trust_store.trust_store();
        if now_unix_s < store.issued_at_unix_s
            || now_unix_s >= store.expires_at_unix_s
        {
            return Err(ReviewVerificationError::TrustSnapshotStale);
        }
        let digest = trust_store.canonical_digest();
        if let Some(highest) = self.last_sequence {
            if trust_store.sequence() < highest {
                return Err(ReviewVerificationError::TrustSnapshotRollback {
                    highest_sequence: highest,
                    received_sequence: trust_store.sequence(),
                });
            }
            if trust_store.sequence() == highest
                && self.last_digest.as_deref() != Some(digest.as_str())
            {
                return Err(ReviewVerificationError::TrustSnapshotSequenceReused {
                    sequence: highest,
                });
            }
        }
        Ok(())
    }

    /// Commit a trust snapshot only after a receipt verified successfully.
    fn record_verified(&mut self, trust_store: &VerifiedReviewTrustStore) {
        self.last_sequence = Some(trust_store.sequence());
        self.last_digest = Some(trust_store.canonical_digest());
    }
}

/// A receipt which passed byte-digest, certificate-binding, lifecycle, freshness,
/// and strict Ed25519 checks against the supplied trust store.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedCalibrationReviewReceipt {
    artifact: VerifiedArtifactBytes,
    receipt: SignedCalibrationReviewReceipt,
    verified_at_unix_s: u64,
    reviewer_authorization: Option<VerifiedReviewerAuthorization>,
}

impl VerifiedCalibrationReviewReceipt {
    /// Verify canonical receipt bytes and the strict Ed25519 signature only.
    ///
    /// This generic method is not sufficient for calibration evidence. For that,
    /// use verify_for_calibration with a provider-specific conflict policy.
    pub fn verify(
        artifact: VerifiedArtifactBytes,
        trust_store: &VerifiedReviewTrustStore,
        trust_store_guard: &mut ReviewTrustStoreGuard,
        expected_record_id: &str,
        expected_certificate: &ArtifactReference,
        now_unix_s: u64,
    ) -> Result<Self, ReviewVerificationError> {
        Self::verify_internal(
            artifact,
            trust_store,
            trust_store_guard,
            expected_record_id,
            expected_certificate,
            None,
            now_unix_s,
        )
    }

    /// Verify the receipt and require root-authorized reviewer identity/organization
    /// metadata with organizational separation from the calibration provider.
    pub fn verify_for_calibration(
        artifact: VerifiedArtifactBytes,
        trust_store: &VerifiedReviewTrustStore,
        trust_store_guard: &mut ReviewTrustStoreGuard,
        expected_record_id: &str,
        expected_certificate: &ArtifactReference,
        authorization_policy: &ReviewAuthorizationPolicy,
        now_unix_s: u64,
    ) -> Result<Self, ReviewVerificationError> {
        Self::verify_internal(
            artifact,
            trust_store,
            trust_store_guard,
            expected_record_id,
            expected_certificate,
            Some(authorization_policy),
            now_unix_s,
        )
    }

    fn verify_internal(
        artifact: VerifiedArtifactBytes,
        trust_store: &VerifiedReviewTrustStore,
        trust_store_guard: &mut ReviewTrustStoreGuard,
        expected_record_id: &str,
        expected_certificate: &ArtifactReference,
        authorization_policy: Option<&ReviewAuthorizationPolicy>,
        now_unix_s: u64,
    ) -> Result<Self, ReviewVerificationError> {
        if artifact.byte_length() == 0 {
            return Err(ReviewVerificationError::EmptyReceipt);
        }
        let receipt = SignedCalibrationReviewReceipt::from_canonical_bytes(artifact.bytes())?;
        trust_store_guard.check_candidate(trust_store, now_unix_s)?;
        trust_store.trust_store().verify(
            &receipt,
            expected_record_id,
            expected_certificate,
            now_unix_s,
        )?;
        let reviewer_authorization = authorization_policy
            .map(|policy| {
                trust_store.trust_store().authorize_reviewer(
                    receipt.payload().key_id(),
                    policy,
                    trust_store.sequence(),
                    trust_store.canonical_digest(),
                    trust_store.authority_key_sha256().to_owned(),
                    now_unix_s,
                )
            })
            .transpose()?;
        // Rejected signatures or authorization decisions must not advance rollback state.
        trust_store_guard.record_verified(trust_store);
        Ok(Self {
            artifact,
            receipt,
            verified_at_unix_s: now_unix_s,
            reviewer_authorization,
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

    pub fn reviewer_authorization(&self) -> Option<&VerifiedReviewerAuthorization> {
        self.reviewer_authorization.as_ref()
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
    InvalidTrustSnapshotSignature,
    TrustSnapshotRollback {
        highest_sequence: u64,
        received_sequence: u64,
    },
    TrustSnapshotSequenceReused {
        sequence: u64,
    },
    InvalidTrustCheckpoint,
    ReviewerNotAuthorized(String),
    ReviewerNotIndependent {
        reviewer_organization_id: String,
        conflicted_organization_id: String,
    },
    TooManyExcludedOrganizations {
        actual: usize,
        maximum: usize,
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
                .unwrap()
                .with_reviewer_authorization(
                    "reviewer-person-a",
                    "independent-calibration-lab",
                )
                .unwrap(),
            ],
        )
        .unwrap()
    }

    fn authority_signing_key() -> SigningKey {
        SigningKey::from_bytes(&[0x24; 32])
    }

    fn verified_trust_store_from(store: ReviewTrustStore) -> VerifiedReviewTrustStore {
        let authority = authority_signing_key();
        let message = SignedReviewTrustStoreSnapshot::signing_message(&store);
        let signature = authority.sign(&message).to_bytes();
        SignedReviewTrustStoreSnapshot::new(store, signature)
            .verify(&authority.verifying_key())
            .unwrap()
    }

    fn verified_trust_store() -> VerifiedReviewTrustStore {
        verified_trust_store_from(trust_store())
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
    fn trust_store_canonical_bytes_roundtrip_and_reject_trailing_data() {
        let store = trust_store();
        let bytes = store.canonical_bytes();
        let parsed = ReviewTrustStore::from_canonical_bytes(&bytes).unwrap();
        assert_eq!(parsed.canonical_bytes(), bytes);
        assert_eq!(parsed.sequence(), store.sequence());

        let mut trailing = bytes;
        trailing.push(0);
        assert_eq!(
            ReviewTrustStore::from_canonical_bytes(&trailing).unwrap_err(),
            ReviewVerificationError::NonCanonicalEncoding
        );
    }

    #[test]
    fn signed_trust_snapshot_roundtrips_and_rejects_mutation() {
        let store = trust_store();
        let authority = authority_signing_key();
        let signature = authority
            .sign(&SignedReviewTrustStoreSnapshot::signing_message(&store))
            .to_bytes();
        let snapshot = SignedReviewTrustStoreSnapshot::new(store, signature);
        let bytes = snapshot.to_canonical_bytes();
        let parsed = SignedReviewTrustStoreSnapshot::from_canonical_bytes(&bytes).unwrap();
        let verified = parsed.verify(&authority.verifying_key()).unwrap();
        assert_eq!(verified.sequence(), TRUST_SEQUENCE);

        let mut tampered = bytes.clone();
        let sequence_offset =
            b"SYMRTRSS2\0".len() + b"SYMRTRS2\0".len() + 2
                + REVIEW_TRUST_STORE_SCHEMA.len();
        tampered[sequence_offset] ^= 1;
        let parsed_tampered = SignedReviewTrustStoreSnapshot::from_canonical_bytes(&tampered)
            .unwrap();
        assert_eq!(
            parsed_tampered.verify(&authority.verifying_key()).unwrap_err(),
            ReviewVerificationError::InvalidTrustSnapshotSignature
        );

        tampered.push(0);
        assert!(
            SignedReviewTrustStoreSnapshot::from_canonical_bytes(&tampered).is_err()
        );
    }

    #[test]
    fn signed_trust_snapshot_authenticates_canonical_key_lifecycle_state() {
        let store = trust_store();
        let authority = authority_signing_key();
        let signature = authority
            .sign(&SignedReviewTrustStoreSnapshot::signing_message(&store))
            .to_bytes();
        let verified = SignedReviewTrustStoreSnapshot::new(store.clone(), signature)
            .verify(&authority.verifying_key())
            .unwrap();

        assert_eq!(verified.sequence(), TRUST_SEQUENCE);
        assert_eq!(
            verified.authority_key_sha256(),
            crate::sha256_hex(authority.verifying_key().as_bytes())
        );
        assert_eq!(
            verified.trust_store().canonical_digest(),
            store.canonical_digest()
        );
    }

    #[test]
    fn signed_trust_snapshot_rejects_tampering_or_wrong_authority() {
        let store = trust_store();
        let authority = authority_signing_key();
        let signature = authority
            .sign(&SignedReviewTrustStoreSnapshot::signing_message(&store))
            .to_bytes();
        let other_authority = SigningKey::from_bytes(&[0x25; 32]);
        assert_eq!(
            SignedReviewTrustStoreSnapshot::new(store.clone(), signature)
                .verify(&other_authority.verifying_key())
                .unwrap_err(),
            ReviewVerificationError::InvalidTrustSnapshotSignature
        );

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
        assert_eq!(
            SignedReviewTrustStoreSnapshot::new(replacement, signature)
                .verify(&authority.verifying_key())
                .unwrap_err(),
            ReviewVerificationError::InvalidTrustSnapshotSignature
        );
    }

    #[test]
    fn canonical_receipt_roundtrips_and_verifies_with_the_configured_trust_key() {
        let receipt = signed_receipt();
        let bytes = receipt.to_canonical_bytes();
        let parsed = SignedCalibrationReviewReceipt::from_canonical_bytes(&bytes).unwrap();
        assert_eq!(parsed, receipt);

        let trust = verified_trust_store();
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

        let trust = verified_trust_store();
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
    fn invalid_signature_does_not_advance_trust_store_guard() {
        let mut receipt = signed_receipt();
        receipt.signature[0] ^= 1;
        let artifact = verified_artifact(&receipt);
        let trust = verified_trust_store();
        let mut guard = ReviewTrustStoreGuard::default();

        assert_eq!(
            VerifiedCalibrationReviewReceipt::verify(
                artifact,
                &trust,
                &mut guard,
                "calibration-record-1",
                &certificate(),
                NOW,
            )
            .unwrap_err(),
            ReviewVerificationError::InvalidSignature
        );
        assert_eq!(guard.last_sequence(), None);
        assert_eq!(guard.checkpoint(), None);
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
        let initial = verified_trust_store();
        let advanced = verified_trust_store_from(
            ReviewTrustStore::new(
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
            .unwrap(),
        );
        let mut guard = ReviewTrustStoreGuard::default();

        guard.check_candidate(&initial, NOW).unwrap();
        guard.record_verified(&initial);
        guard.check_candidate(&initial, NOW).unwrap();
        guard.record_verified(&initial);
        guard.check_candidate(&advanced, NOW).unwrap();
        guard.record_verified(&advanced);
        assert_eq!(guard.last_sequence(), Some(TRUST_SEQUENCE + 1));
    }

    #[test]
    fn trust_store_guard_rejects_sequence_rollback() {
        let newer = verified_trust_store_from(
            ReviewTrustStore::new(
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
            .unwrap(),
        );
        let mut guard = ReviewTrustStoreGuard::default();
        guard.check_candidate(&newer, NOW).unwrap();
        guard.record_verified(&newer);

        assert_eq!(
            guard.check_candidate(&verified_trust_store(), NOW),
            Err(ReviewVerificationError::TrustSnapshotRollback {
                highest_sequence: TRUST_SEQUENCE + 1,
                received_sequence: TRUST_SEQUENCE,
            })
        );
    }

    #[test]
    fn authenticated_checkpoint_restores_rollback_high_water_mark() {
        let newer = verified_trust_store_from(
            ReviewTrustStore::new(
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
            .unwrap(),
        );
        let mut guard = ReviewTrustStoreGuard::default();
        guard.check_candidate(&newer, NOW).unwrap();
        guard.record_verified(&newer);
        let (sequence, digest) = guard.checkpoint().unwrap();
        assert_eq!(sequence, TRUST_SEQUENCE + 1);
        assert_eq!(guard.last_digest(), Some(digest.as_str()));

        let mut restored = ReviewTrustStoreGuard::from_checkpoint(sequence, digest).unwrap();
        assert_eq!(
            restored.check_candidate(&verified_trust_store(), NOW),
            Err(ReviewVerificationError::TrustSnapshotRollback {
                highest_sequence: TRUST_SEQUENCE + 1,
                received_sequence: TRUST_SEQUENCE,
            })
        );
        assert_eq!(
            ReviewTrustStoreGuard::from_checkpoint(0, "a".repeat(64)).unwrap_err(),
            ReviewVerificationError::InvalidTrustCheckpoint
        );
        assert_eq!(
            ReviewTrustStoreGuard::from_checkpoint(1, "A".repeat(64)).unwrap_err(),
            ReviewVerificationError::InvalidDigest
        );
    }

    #[test]
    fn trust_store_guard_rejects_different_contents_at_the_same_sequence() {
        let replacement = verified_trust_store_from(
            ReviewTrustStore::new(
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
            .unwrap(),
        );
        let initial = verified_trust_store();
        let mut guard = ReviewTrustStoreGuard::default();
        guard.check_candidate(&initial, NOW).unwrap();
        guard.record_verified(&initial);

        assert_eq!(
            guard.check_candidate(&replacement, NOW),
            Err(ReviewVerificationError::TrustSnapshotSequenceReused {
                sequence: TRUST_SEQUENCE,
            })
        );
    }

    #[test]
    fn trust_snapshot_and_receipt_expiration_are_enforced() {
        let receipt = signed_receipt();
        assert_eq!(
            verified_trust_store()
                .trust_store()
                .verify(&receipt, "calibration-record-1", &certificate(), NOW + 100),
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
                &verified_trust_store(),
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
    
    #[test]
    fn calibration_review_requires_root_authorized_independent_reviewer_metadata() {
        let receipt = signed_receipt();
        let trust = verified_trust_store();
        let policy = ReviewAuthorizationPolicy::new(
            "calibration-provider-a",
            vec!["conflict-of-interest-org".to_owned()],
        )
        .unwrap();
        let mut guard = ReviewTrustStoreGuard::default();
        let verified = VerifiedCalibrationReviewReceipt::verify_for_calibration(
            verified_artifact(&receipt),
            &trust,
            &mut guard,
            "calibration-record-1",
            &certificate(),
            &policy,
            NOW,
        )
        .unwrap();
        let authorization = verified
            .reviewer_authorization()
            .expect("calibration approval must carry authorization proof");
        assert_eq!(authorization.reviewer_id(), "reviewer-person-a");
        assert_eq!(
            authorization.reviewer_organization_id(),
            "independent-calibration-lab"
        );
        assert_eq!(
            authorization.provider_organization_id(),
            "calibration-provider-a"
        );
        assert_eq!(authorization.policy_sha256().len(), 64);
        assert_eq!(authorization.trust_store_sequence(), TRUST_SEQUENCE);
        assert_eq!(authorization.trust_store_sha256().len(), 64);
        assert_eq!(authorization.trust_authority_key_sha256().len(), 64);
        assert_eq!(authorization.verified_at_unix_s(), NOW);
        assert_eq!(
            authorization.excluded_organization_ids().collect::<Vec<_>>(),
            vec!["conflict-of-interest-org"]
        );
        assert_eq!(guard.last_sequence(), Some(TRUST_SEQUENCE));
    }

    #[test]
    fn conflicted_or_unidentified_reviewers_fail_without_advancing_guard() {
        let receipt = signed_receipt();
        let trust = verified_trust_store();

        for policy in [
            ReviewAuthorizationPolicy::new("independent-calibration-lab", Vec::new()).unwrap(),
            ReviewAuthorizationPolicy::new(
                "calibration-provider-a",
                vec!["independent-calibration-lab".to_owned()],
            )
            .unwrap(),
        ] {
            let mut guard = ReviewTrustStoreGuard::default();
            assert!(matches!(
                VerifiedCalibrationReviewReceipt::verify_for_calibration(
                    verified_artifact(&receipt),
                    &trust,
                    &mut guard,
                    "calibration-record-1",
                    &certificate(),
                    &policy,
                    NOW,
                ),
                Err(ReviewVerificationError::ReviewerNotIndependent { .. })
            ));
            assert_eq!(guard.last_sequence(), None);
            assert_eq!(guard.checkpoint(), None);
        }

        let unauthorised_store = ReviewTrustStore::new(
            TRUST_SEQUENCE,
            NOW - 10,
            NOW + 100,
            vec![ReviewTrustKey::new(
                "reviewer-a",
                signing_key().verifying_key().to_bytes(),
                NOW - 100,
                Some(NOW + 500),
                None,
            )
            .unwrap()],
        )
        .unwrap();
        let unauthorised_trust = verified_trust_store_from(unauthorised_store);
        let mut guard = ReviewTrustStoreGuard::default();
        let policy =
            ReviewAuthorizationPolicy::new("calibration-provider-a", Vec::new()).unwrap();
        assert_eq!(
            VerifiedCalibrationReviewReceipt::verify_for_calibration(
                verified_artifact(&receipt),
                &unauthorised_trust,
                &mut guard,
                "calibration-record-1",
                &certificate(),
                &policy,
                NOW,
            )
            .unwrap_err(),
            ReviewVerificationError::ReviewerNotAuthorized("reviewer-a".to_owned())
        );
        assert_eq!(guard.last_sequence(), None);
    }

}
